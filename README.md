# PDF Invoice OCR Parser

A **Streamlit-based web application** that extracts structured data from PDF invoices using OCR powered by **Google's Gemini API**. The application processes PDF invoices, converts them to images, and extracts key information such as invoice numbers, dates, line items, and totals, presenting the results in a user-friendly interface. The extracted data can be downloaded as a JSON file but is **not saved to disk**.

---

## 🚀 Features

- 📤 Upload PDF invoices and extract structured data in **JSON format**.
- 🔍 Supports extraction of key fields including:
  - Invoice Number
  - Invoice Date
  - Due Date
  - Vendor/Customer Details
  - Line Items
  - Totals
- ✍️ Handles both digital and **handwritten text** with rules for Purchase Order (PO) number extraction.
- 📊 Tracks and displays **token usage** for the Gemini API.
- 📥 Provides a **downloadable JSON file** with the extracted data.
- ⚙️ Configurable via `config.yaml` and environment variables.
- 🧾 **Robust logging** to console and file for debugging and monitoring.

---

## ✅ Prerequisites

- Python 3.8 or higher
- A valid **Google Gemini API key** (set via `.env` or UI)
- Required Python packages (`requirements.txt`)

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
````

### 2. Set Up a Virtual Environment (Recommended)

```bash
python -m venv venv
# On Linux/macOS
source venv/bin/activate
# On Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory:

```
GEMINI_API_KEY=your-google-gemini-api-key
```

### 5. Configure the Application

Ensure a `config.yaml` file exists with the following:

```yaml
app:
  title: "PDF Invoice OCR Parser"
  icon: "📄"
  description: |
    Upload a PDF invoice to extract structured data using OCR powered by Google's Gemini API.
    The system will analyze the invoice and extract key information like invoice numbers, dates, line items, and totals.
  about: |
    This application uses OCR and AI to extract structured data from invoice PDFs. Powered by Google's Gemini API and Streamlit.

gemini:
  model: "gemini-2.0-flash"

processing:
  default_dpi: 300
  dpi_min: 150
  dpi_max: 600
  dpi_step: 50
```

---

## ▶️ Usage

### Run the Application

```bash
streamlit run main.py
```

By default, Streamlit will start at: [http://localhost:8501](http://localhost:8501)

### Interact with the UI:

* Open the app in your browser.
* Optionally input your API key in the sidebar (if not in `.env`).
* Adjust DPI settings if needed.
* Upload your PDF invoice.
* View extracted structured data.
* Download data as a `.json` file.
* Monitor API token usage.

---

## 📤 Output

* Extracted data is **not saved to disk**.
* A downloadable **JSON file** is generated via UI.
* Logs are written to `app.log`.

---

## ⚙️ Configuration

### Environment Variables

* `GEMINI_API_KEY`: Your Google Gemini API key (required).

### Config File: `config.yaml`

* `app`: UI details
* `gemini`: Gemini model settings
* `processing`: DPI values

---

## 📚 Logging

* Logs output to **console** and `app.log`
* Log levels:

  * INFO
  * DEBUG
  * WARNING
  * ERROR
  * CRITICAL

---

## ❗ Error Handling

* File upload & parsing errors are shown in the UI and logged.
* API and JSON errors are captured and stack-traced.
* Temporary files are auto-cleaned, even on failure.

---

## 🔐 Security Notes

* Store sensitive API keys in `.env` only.
* UI input for API key is masked like a password.

---

## 🛠 Development

* **Dependencies**: Use `requirements.txt`
* **Testing**: Use sample PDF invoices
* **Extending**: Modify `PDFInvoiceOCRParser.extract_invoice_data()` to change prompt/rules

---

## 🧯 Troubleshooting

* **API Key Issues**: Verify `.env` or UI key
* **File Errors**: Ensure uploaded file is a valid PDF
* **JSON Issues**: Check logs (`app.log`) for response errors
* **Memory Issues**: Lower the DPI in config

---
