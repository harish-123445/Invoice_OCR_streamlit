import os
import json
import tempfile
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import streamlit as st
import fitz
import google.generativeai as genai
from PIL import Image
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')  # Log to file for production
    ]
)
logger = logging.getLogger(__name__)

# Load configuration from config.yaml
def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        raise

config = load_config()

class PDFInvoiceOCRParser:
    """
    A class to parse PDF invoices using OCR and extract structured data with Google's Gemini API.
    Also tracks token usage from the API.
    """
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        """
        Initialize the PDF parser.
        
        Args:
            gemini_api_key: Google Gemini API key. If None, will try to read from environment variable.
        """
        self.api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass it to the constructor.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(config['gemini']['model'])
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.pages_processed = 0
        logger.info("PDFInvoiceOCRParser initialized successfully")
    
    def pdf_to_images(self, pdf_path: str, dpi: int = None) -> List[Image.Image]:
        """
        Convert each page of a PDF file to an image.
        
        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for the converted images
            
        Returns:
            List of PIL Image objects
        """
        dpi = dpi or config['processing']['default_dpi']
        logger.info(f"Converting PDF to images: {pdf_path}")
        
        pdf_document = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            images.append(img)
            logger.debug(f"Converted page {page_num + 1} to image")
        
        logger.info(f"PDF conversion complete. Generated {len(images)} images.")
        return images
    
    def extract_invoice_data(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract structured data from an invoice image using Google's Gemini API.
        Also tracks token usage from the API.
        
        Args:
            image: PIL Image object of the invoice
            
        Returns:
            Dictionary with extracted key-value pairs from the invoice
        """
        logger.info("Extracting invoice data from image")
        
        prompt = """
Please analyze the provided invoice image using OCR technology and extract the following information in a structured JSON format:

Required Fields
- Invoice Number
- Invoice Date
- Due Date
- Vendor Name
- Vendor Address
- Customer/Client Name
- Customer/Client Address
- Line Items (including quantity, description, unit price, tax price, and total price)
- Subtotal
- Tax Amount
- Total Amount Due
- Payment Terms
- Payment Method (if available)
- Sales Order Number (if available)
- Buyer Order Number (if available)
- Purchase Order Number (also labeled as PO Number)

PO Number Extraction Rules
- DO NOT take "NO" as PO number, Remove the value from PO Number field.
- If a Date is found in the "PO Number" column, Remove the value date.
- PO numbers manually entered (handwritten) should be prioritized.
- Do not fetch the PO number based on the Sales Order Number.
- PO numbers are not a floating point number, so do not include any decimal points.
- Extract all PO numbers, whether digital or handwritten. If multiple PO numbers exist, return them as a list.
- Do NOT include any value that resembles a date in the list of PO numbers.
- Specifically, exclude any values that match common date formats such as:
        - dd.mm.yyyy
        - dd/mm/yyyy
        - yyyy-mm-dd
        - mm-dd-yyyy
        - dd-mm-yy

Special Instructions
- The invoice may contain both digital and handwritten text - extract both. 
- Return results in properly formatted JSON.
- For any field not found in the image, set the value to null.
- Don't map the same value to multiple fields.
- If a field is not applicable, set it to null.
- Invoice URL (if available)

Please ensure all relevant information is accurately extracted, regardless of format or placement within the invoice.
"""
        
        try:
            response = self.model.generate_content([prompt, image])
            
            try:
                if hasattr(response, 'usage_metadata'):
                    if hasattr(response.usage_metadata, 'prompt_token_count'):
                        self.total_input_tokens += response.usage_metadata.prompt_token_count
                    elif hasattr(response.usage_metadata, 'prompt_tokens'):
                        self.total_input_tokens += response.usage_metadata.prompt_tokens
                    
                    if hasattr(response.usage_metadata, 'candidates_token_count'):
                        self.total_output_tokens += response.usage_metadata.candidates_token_count
                    elif hasattr(response.usage_metadata, 'completion_tokens'):
                        self.total_output_tokens += response.usage_metadata.completion_tokens
                    elif hasattr(response.usage_metadata, 'response_tokens'):
                        self.total_output_tokens += response.usage_metadata.response_tokens
                elif hasattr(response, 'tokens'):
                    if hasattr(response.tokens, 'input'):
                        self.total_input_tokens += response.tokens.input
                    if hasattr(response.tokens, 'output'):
                        self.total_output_tokens += response.tokens.output
                
                self.pages_processed += 1
                
            except Exception as token_err:
                logger.warning(f"Could not track token usage: {token_err}")
            
            response_text = response.text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = response_text[json_start:json_end]
                try:
                    invoice_data = json.loads(json_content)
                    logger.info("Successfully extracted and parsed invoice data")
                    return invoice_data
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from Gemini response: {e}")
                    return {"error": "Failed to parse JSON from API response", "raw_response": response_text}
            else:
                logger.warning("No JSON content found in Gemini response")
                return {"error": "No JSON content found in API response", "raw_response": response_text}
                
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return {"error": f"API call failed: {str(e)}"}
    
    def process_invoice(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a PDF invoice: convert to images and extract data from each page.
        
        Args:
            pdf_path: Path to the PDF invoice file
            
        Returns:
            Dictionary containing extracted data from all pages and token usage info
        """
        logger.info(f"Processing invoice: {pdf_path}")
        
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.pages_processed = 0
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return {"status": "error", "message": f"PDF file not found: {pdf_path}"}
        
        try:
            images = self.pdf_to_images(pdf_path)
            results = []
            for i, image in enumerate(images):
                logger.info(f"Processing page {i+1} of {len(images)}")
                page_data = self.extract_invoice_data(image)
                page_data["page_number"] = i + 1
                page_data["total_pages"] = len(images)
                results.append(page_data)
            
            token_summary = {
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
                "pages_processed": self.pages_processed,
                "tokens_per_page": {
                    "input": self.total_input_tokens / max(1, self.pages_processed),
                    "output": self.total_output_tokens / max(1, self.pages_processed),
                    "total": (self.total_input_tokens + self.total_output_tokens) / max(1, self.pages_processed)
                }
            }
            
            logger.info(f"Invoice processing complete. Extracted data from {len(results)} pages.")
            return {"status": "success", "data": results, "token_usage": token_summary}
            
        except Exception as e:
            logger.error(f"Error processing invoice: {e}")
            return {"status": "error", "message": f"Failed to process invoice: {str(e)}"}
    
    def process_and_save(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a PDF invoice and return the extracted data without saving to a file.
        
        Args:
            pdf_path: Path to the PDF invoice file
            
        Returns:
            Dictionary with extracted data and token usage information
        """
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.pages_processed = 0
        
        result = self.process_invoice(pdf_path)
        
        logger.info("Invoice processing complete. Results not saved to file as per request.")
        return {
            "token_usage": result.get("token_usage", {}),
            "results": result.get("data", [])
        }

def main():
    # Set up the page config
    st.set_page_config(
        page_title=config['app']['title'],
        page_icon=config['app']['icon'],
        layout="wide"
    )

    # Page title and description
    st.title(config['app']['title'])
    st.markdown(config['app']['description'])
    
    # Sidebar for API key input and options
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            placeholder="Enter your API key",
            help="Enter your Google Gemini API key. If left blank, the default key from environment will be used."
        )
        
        dpi_option = st.slider(
            "Image DPI",
            min_value=config['processing']['dpi_min'],
            max_value=config['processing']['dpi_max'],
            value=config['processing']['default_dpi'],
            step=config['processing']['dpi_step'],
            help="Higher DPI may improve text recognition but uses more memory"
        )
        
        st.divider()
        st.header("About")
        st.info(config['app']['about'])

    # File uploader
    uploaded_file = st.file_uploader("Upload an invoice PDF", type=["pdf"])
    
    if uploaded_file is not None:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Uploaded Invoice")
            file_details = {"Filename": uploaded_file.name, "File Size": f"{uploaded_file.size / 1024:.2f} KB"}
            st.json(file_details)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                pdf_path = tmp_file.name
                
            status_text.text("PDF saved. Initializing parser...")
            progress_bar.progress(10)
            
            try:
                parser = PDFInvoiceOCRParser(gemini_api_key=api_key if api_key else None)
                status_text.text("Parser initialized. Processing invoice...")
                progress_bar.progress(20)
                
                with st.spinner("Extracting data from invoice..."):
                    result = parser.process_and_save(pdf_path)
                    progress_bar.progress(90)
                    status_text.text("Extraction complete! Displaying results...")
                
                st.subheader("Token Usage")
                token_usage = result['token_usage']
                
                token_cols = st.columns(4)
                token_cols[0].metric("Input Tokens", token_usage.get('total_input_tokens', 0))
                token_cols[1].metric("Output Tokens", token_usage.get('total_output_tokens', 0))
                token_cols[2].metric("Total Tokens", token_usage.get('total_tokens', 0))
                token_cols[3].metric("Pages Processed", token_usage.get('pages_processed', 0))
                
                if token_usage.get('pages_processed', 0) > 0:
                    st.subheader("Per Page Analysis")
                    per_page_cols = st.columns(3)
                    per_page_cols[0].metric("Avg Input Tokens/Page", f"{token_usage['tokens_per_page']['input']:.2f}")
                    per_page_cols[1].metric("Avg Output Tokens/Page", f"{token_usage['tokens_per_page']['output']:.2f}")
                    per_page_cols[2].metric("Avg Total Tokens/Page", f"{token_usage['tokens_per_page']['total']:.2f}")
                
                # Create JSON data for download without saving to file
                json_data = json.dumps(result, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"{Path(uploaded_file.name).stem}_invoice_data.json",
                    mime="application/json"
                )
                
                progress_bar.progress(100)
                
            except Exception as e:
                st.error(f"Error processing the invoice: {str(e)}")
                logger.error(f"Processing error: {e}", exc_info=True)
                if os.path.exists(pdf_path):
                    os.unlink(pdf_path)
                st.stop()
        
        with col2:
            st.subheader("Extracted Invoice Data")
            
            for i, page_data in enumerate(result.get('results', [])):
                if i > 0:
                    st.divider()
                
                st.markdown(f"### Page {page_data.get('page_number', i+1)} of {page_data.get('total_pages', len(result.get('results', [])))}")
                
                if 'error' in page_data:
                    st.error(f"Error: {page_data['error']}")
                    continue
                
                with st.expander("Invoice Details", expanded=True):
                    info_cols = st.columns(3)
                    info_cols[0].markdown("**Invoice Number**")
                    info_cols[0].write(page_data.get('Invoice Number', 'N/A'))
                    info_cols[1].markdown("**Invoice Date**")
                    info_cols[1].write(page_data.get('Invoice Date', 'N/A'))
                    info_cols[2].markdown("**Due Date**")
                    info_cols[2].write(page_data.get('Due Date', 'N/A'))
                
                with st.expander("Vendor & Customer Information", expanded=True):
                    vendor_customer_cols = st.columns(2)
                    with vendor_customer_cols[0]:
                        st.markdown("**Vendor Details**")
                        st.write(f"**Name:** {page_data.get('Vendor Name', 'N/A')}")
                        st.write(f"**Address:** {page_data.get('Vendor Address', 'N/A')}")
                    with vendor_customer_cols[1]:
                        st.markdown("**Customer Details**")
                        st.write(f"**Name:** {page_data.get('Customer/Client Name', 'N/A')}")
                        st.write(f"**Address:** {page_data.get('Customer/Client Address', 'N/A')}")
                
                with st.expander("Line Items", expanded=True):
                    line_items = page_data.get('Line Items', [])
                    if line_items and isinstance(line_items, list) and len(line_items) > 0:
                        import pandas as pd
                        try:
                            df = pd.DataFrame(line_items)
                            st.dataframe(df, use_container_width=True)
                        except Exception as e:
                            st.write(line_items)
                    else:
                        st.write("No line items found or unable to parse line items.")
                
                with st.expander("Financial Summary", expanded=True):
                    financial_cols = st.columns(4)
                    financial_cols[0].markdown("**Subtotal**")
                    financial_cols[0].write(page_data.get('Subtotal', 'N/A'))
                    financial_cols[1].markdown("**Tax Amount**")
                    financial_cols[1].write(page_data.get('Tax Amount', 'N/A'))
                    financial_cols[2].markdown("**Total Amount Due**")
                    financial_cols[2].write(page_data.get('Total Amount Due', 'N/A'))
                    financial_cols[3].markdown("**Payment Terms**")
                    financial_cols[3].write(page_data.get('Payment Terms', 'N/A'))
                
                with st.expander("Additional Information", expanded=False):
                    st.markdown("**Payment Method**")
                    st.write(page_data.get('Payment Method', 'N/A'))
                    st.markdown("**Sales Order Number**")
                    st.write(page_data.get('Sales Order Number', 'N/A'))
                    st.markdown("**Buyer Order Number**")
                    st.write(page_data.get('Buyer Order Number', 'N/A'))
                    st.markdown("**Purchase Order Number**")
                    st.write(page_data.get('Purchase Order Number', 'N/A'))
                    st.markdown("**Invoice URL**")
                    st.write(page_data.get('Invoice URL', 'N/A'))
                
                with st.expander("Raw JSON Data", expanded=False):
                    st.json(page_data)
            
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application failed to start: {e}", exc_info=True)
        st.error("Application failed to start. Please check the logs for details.")