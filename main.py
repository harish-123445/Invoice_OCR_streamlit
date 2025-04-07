import os
import json
import base64
from typing import List, Dict, Any, Optional
import tempfile
import logging
from pathlib import Path
import streamlit as st

import fitz  
import google.generativeai as genai
from PIL import Image
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        # Get API key from parameter or environment variable
        self.api_key = gemini_api_key or "AIzaSyA9I36rGn5QFV-GP8YyUvmBAzQaOsyELi4"
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass it to the constructor.")
        
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
        # Get the Gemini Vision model for image processing
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.pages_processed = 0
        
        logger.info("PDFInvoiceOCRParser initialized successfully")
    
    def pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List[Image.Image]:
        """
        Convert each page of a PDF file to an image.
        
        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for the converted images
            
        Returns:
            List of PIL Image objects
        """
        logger.info(f"Converting PDF to images: {pdf_path}")
        
        # Open the PDF file
        pdf_document = fitz.open(pdf_path)
        images = []
        
        # Iterate through each page
        for page_num in range(len(pdf_document)):
            # Get the page
            page = pdf_document[page_num]
            
            # Convert page to a pixmap (image)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            
            # Convert pixmap to PIL Image
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
        
        # Prepare the prompt for Gemini
        prompt = """
        Analyze this invoice image using OCR and extract the following information in a structured format:
        - Invoice Number
        - Invoice Date
        - Due Date
        - Vendor Name
        - Vendor Address
        - Customer/Client Name
        - Customer/Client Address
        - Line Items (with quantity, description, unit price, Tax Price and total price)
        - Subtotal
        - Tax Amount
        - Total Amount Due
        - Payment Terms
        - Payment Method (if available)
     
        
        Return the results in a clean JSON format with these fields. If any field is not found in the image, 
        set its value to null. Make sure the JSON is properly formatted and valid.
        """
        
        try:
            # Call Gemini API with the image
            response = self.model.generate_content([prompt, image])
            
            # Track token usage if available in response
            # The attribute names may vary depending on the API version
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
                        
                # Try alternate response structure if available
                elif hasattr(response, 'tokens'):
                    if hasattr(response.tokens, 'input'):
                        self.total_input_tokens += response.tokens.input
                    if hasattr(response.tokens, 'output'):
                        self.total_output_tokens += response.tokens.output
                
                # Increment pages processed
                self.pages_processed += 1
                
            except Exception as token_err:
                logger.warning(f"Could not track token usage: {token_err}")
            
            # Extract JSON from the response
            response_text = response.text
            
            # Find JSON content in the response (in case it's wrapped in markdown code blocks)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = response_text[json_start:json_end]
                try:
                    # Parse the JSON data
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
    
    def process_invoice(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Process a PDF invoice: convert to images and extract data from each page.
        
        Args:
            pdf_path: Path to the PDF invoice file
            
        Returns:
            List of dictionaries containing extracted data from each page
        """
        logger.info(f"Processing invoice: {pdf_path}")
        
        # Check if file exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return [{"error": f"PDF file not found: {pdf_path}"}]
        
        try:
            # Convert PDF to images
            images = self.pdf_to_images(pdf_path)
            
            # Process each image
            results = []
            for i, image in enumerate(images):
                logger.info(f"Processing page {i+1} of {len(images)}")
                page_data = self.extract_invoice_data(image)
                
                # Add page information
                page_data["page_number"] = i + 1
                page_data["total_pages"] = len(images)
                
                results.append(page_data)
            
            logger.info(f"Invoice processing complete. Extracted data from {len(results)} pages.")
            return results
            
        except Exception as e:
            logger.error(f"Error processing invoice: {e}")
            return [{"error": f"Failed to process invoice: {str(e)}"}]
    
    def process_and_save(self, pdf_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a PDF invoice and save the results to a JSON file.
        
        Args:
            pdf_path: Path to the PDF invoice file
            output_path: Path to save the output JSON file. If None, will use the PDF name with .json extension
            
        Returns:
            Dictionary with output path and token usage information
        """
        # Reset token counters for this processing run
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.pages_processed = 0
        
        # Generate output path if not provided
        if output_path is None:
            pdf_name = Path(pdf_path).stem
            output_path = f"{pdf_name}_invoice_data.json"
        elif os.path.isdir(output_path) or output_path.endswith('/') or output_path.endswith('\\'):
            # If output_path is a directory, create a file within it
            os.makedirs(output_path, exist_ok=True)
            pdf_name = Path(pdf_path).stem
            output_path = os.path.join(output_path, f"{pdf_name}_invoice_data.json")
        
        # Process the invoice
        results = self.process_invoice(pdf_path)
        
        # Create token usage summary
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
        
        # Combine results and token information
        output_data = {
            "results": results,
            "token_usage": token_summary
        }
        
        # Make sure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save results to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved invoice data to {output_path}")
        logger.info(f"Total tokens used: Input: {self.total_input_tokens}, Output: {self.total_output_tokens}")
        
        return {
            "output_path": output_path,
            "token_usage": token_summary,
            "results": results  # Return the results directly as well
        }

    def get_token_usage(self) -> Dict[str, Any]:
        """
        Get the current token usage statistics.
        
        Returns:
            Dictionary with token usage information
        """
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "pages_processed": self.pages_processed,
            "average_tokens_per_page": {
                "input": self.total_input_tokens / max(1, self.pages_processed),
                "output": self.total_output_tokens / max(1, self.pages_processed),
                "total": (self.total_input_tokens + self.total_output_tokens) / max(1, self.pages_processed)
            }
        }

# Streamlit UI implementation
def main():
    # Set up the page config
    st.set_page_config(
        page_title="Invoice OCR Parser",
        page_icon="📄",
        layout="wide"
    )

    # Page title and description
    st.title("PDF Invoice OCR Parser")
    st.markdown("""
    Upload a PDF invoice to extract structured data using OCR powered by Google's Gemini API.
    The system will analyze the invoice and extract key information like invoice numbers, dates, line items, and totals.
    """)
    
    # Sidebar for API key input and options
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Google Gemini API Key", 
                               type="password", 
                               placeholder="Enter your API key",
                               help="Enter your Google Gemini API key. If left blank, the default key will be used.")
        
        dpi_option = st.slider("Image DPI", min_value=150, max_value=600, value=300, step=50,
                              help="Higher DPI may improve text recognition but uses more memory")
        
        st.divider()
        st.header("About")
        st.info("This application uses OCR and AI to extract structured data from invoice PDFs. Powered by Google's Gemini API and Streamlit.")

    # File uploader
    uploaded_file = st.file_uploader("Upload an invoice PDF", type=["pdf"])
    
    if uploaded_file is not None:
        # Create a progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Layout with columns
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Uploaded Invoice")
            file_details = {"Filename": uploaded_file.name, "File Size": f"{uploaded_file.size / 1024:.2f} KB"}
            st.json(file_details)
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                pdf_path = tmp_file.name
                
            status_text.text("PDF saved. Initializing parser...")
            progress_bar.progress(10)
            
            # Initialize parser with API key (if provided)
            try:
                parser = PDFInvoiceOCRParser(gemini_api_key=api_key if api_key else None)
                status_text.text("Parser initialized. Processing invoice...")
                progress_bar.progress(20)
                
                # Process the invoice
                with st.spinner("Extracting data from invoice..."):
                    result = parser.process_and_save(pdf_path)
                    progress_bar.progress(90)
                    status_text.text("Extraction complete! Displaying results...")
                
                # Display token usage metrics
                st.subheader("Token Usage")
                token_usage = result['token_usage']
                
                token_cols = st.columns(4)
                token_cols[0].metric("Input Tokens", token_usage['total_input_tokens'])
                token_cols[1].metric("Output Tokens", token_usage['total_output_tokens'])
                token_cols[2].metric("Total Tokens", token_usage['total_tokens'])
                token_cols[3].metric("Pages Processed", token_usage['pages_processed'])
                
                if token_usage['pages_processed'] > 0:
                    st.subheader("Per Page Analysis")
                    per_page_cols = st.columns(3)
                    per_page_cols[0].metric("Avg Input Tokens/Page", f"{token_usage['tokens_per_page']['input']:.2f}")
                    per_page_cols[1].metric("Avg Output Tokens/Page", f"{token_usage['tokens_per_page']['output']:.2f}")
                    per_page_cols[2].metric("Avg Total Tokens/Page", f"{token_usage['tokens_per_page']['total']:.2f}")
                
                # Add download button for the JSON file
                with open(result['output_path'], "r") as f:
                    json_data = f.read()
                
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"{Path(uploaded_file.name).stem}_invoice_data.json",
                    mime="application/json"
                )
                
                progress_bar.progress(100)
                
            except Exception as e:
                st.error(f"Error processing the invoice: {str(e)}")
                if os.path.exists(pdf_path):
                    os.unlink(pdf_path)
                st.stop()
        
        with col2:
            # Display extracted data
            st.subheader("Extracted Invoice Data")
            
            # Loop through each page result
            for i, page_data in enumerate(result.get('results', [])):
                if i > 0:  # Add separator between pages
                    st.divider()
                
                st.markdown(f"### Page {page_data.get('page_number', i+1)} of {page_data.get('total_pages', len(result.get('results', [])))}")
                
                # Check for errors
                if 'error' in page_data:
                    st.error(f"Error: {page_data['error']}")
                    continue
                
                # Create expandable sections for different parts of the invoice
                with st.expander("Invoice Details", expanded=True):
                    info_cols = st.columns(3)
                    
                    # Basic invoice information
                    info_cols[0].markdown("**Invoice Number**")
                    info_cols[0].write(page_data.get('Invoice Number', 'N/A'))
                    
                    info_cols[1].markdown("**Invoice Date**")
                    info_cols[1].write(page_data.get('Invoice Date', 'N/A'))
                    
                    info_cols[2].markdown("**Due Date**")
                    info_cols[2].write(page_data.get('Due Date', 'N/A'))
                
                # Vendor and customer info
                with st.expander("Vendor & Customer Information", expanded=True):
                    vendor_customer_cols = st.columns(2)
                    
                    # Vendor information
                    with vendor_customer_cols[0]:
                        st.markdown("**Vendor Details**")
                        st.write(f"**Name:** {page_data.get('Vendor Name', 'N/A')}")
                        st.write(f"**Address:** {page_data.get('Vendor Address', 'N/A')}")
                    
                    # Customer information
                    with vendor_customer_cols[1]:
                        st.markdown("**Customer Details**")
                        st.write(f"**Name:** {page_data.get('Customer/Client Name', 'N/A')}")
                        st.write(f"**Address:** {page_data.get('Customer/Client Address', 'N/A')}")
                
                # Line items
                with st.expander("Line Items", expanded=True):
                    line_items = page_data.get('Line Items', [])
                    if line_items and isinstance(line_items, list) and len(line_items) > 0:
                        # Convert to DataFrame for better display
                        import pandas as pd
                        try:
                            df = pd.DataFrame(line_items)
                            st.dataframe(df, use_container_width=True)
                        except Exception as e:
                            st.write(line_items)  # Fallback if can't convert to DataFrame
                    else:
                        st.write("No line items found or unable to parse line items.")
                
                # Financial summary
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
                
                # Payment information
                with st.expander("Payment Information", expanded=False):
                    st.markdown("**Payment Method**")
                    st.write(page_data.get('Payment Method', 'N/A'))
                
                # Raw JSON data
                with st.expander("Raw JSON Data", expanded=False):
                    st.json(page_data)
            
            # Clean up temp file
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

if __name__ == "__main__":
    main()