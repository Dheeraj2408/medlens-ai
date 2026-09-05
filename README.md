# MedLens AI — Intelligent Clinical Lab Report Digitizer

> **Disclaimer:** MedLens AI is intended for clinical information organization and workflow digitization only. It does not provide medical diagnoses, treatment advice, or patient-specific clinical management recommendations.

MedLens AI is a clinical document digitization tool powered by the **Google GenAI SDK** (`google-genai`) and Google Gemini models. It extracts test names, measured values, units, and document-printed reference ranges from medical lab reports (PDFs, JPGs, PNGs, and WebPs) with **strict document grounding** and **provenance tracking**.

---

## 🌟 Key Features

* **Strict Document Grounding (Zero Hallucination Policy):**
  * The extraction engine strictly extracts reference intervals printed on the physical document.
  * If a reference interval is missing or unspecified in the report, it is explicitly marked as `"Not stated in report"` and is **never fabricated or assumed** from general medical knowledge.
* **Clinical Provenance Separation:**
  * Clearly distinguishes between **`User-Provided`** patient intake data (symptoms, allergies, demographics) and **`AI-Extracted`** document findings to ensure strict auditability and medical safety.
* **Modern Web Interface:**
  * Intuitive drag-and-drop file uploader with live thumbnail previews and file size indicators.
  * Interactive lab test table with status badges (`Within Range`, `High`, `Low`, `Abnormal`, `Not Stated`).
  * Instant filter pills to isolate flagged or out-of-range test values.
* **Zero-Setup Offline Demo:**
  * Includes a built-in sample lab panel demonstrating extraction fidelity, out-of-range flags, and unstated reference ranges without requiring an API key or consuming quota.
* **Clinical Overview & Exporting:**
  * Side-by-side comparative cards summarizing patient intake vs. lab findings.
  * One-click **Copy Overview** to clipboard and **Print / Save as PDF** support.

---

## 🏗️ Architecture & Project Structure

```text
medlens-ai/
├── app.py              # Lightweight HTTP backend serving static UI and extraction API
├── extractor.py        # Gemini document extractor with strict Pydantic grounding schemas
├── index.html          # Interactive, responsive clinical web application
├── requirements.txt    # Python dependencies (google-genai, pydantic)
├── .gitignore          # Git exclusion rules
└── README.md           # Project overview and setup instructions
```

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10 or higher
* A [Google Gemini API Key](https://aistudio.google.com/) (for live report processing)

### 1. Clone the Repository
```bash
git clone https://github.com/Dheeraj2408/medlens-ai.git
cd medlens-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```
Your default web browser will automatically open to `http://127.0.0.1:8765/`.

*(Alternatively, you can open `index.html` directly in any web browser to test the demo mode completely offline!)*

---

## 🧪 Usage Guide

1. **Patient Intake:** Fill in patient demographics, reported symptoms, and known allergies in the left sidebar.
2. **API Key:** Enter your Gemini API key in the sidebar (or set `GEMINI_API_KEY` in your environment variables).
3. **Upload Report:** Drag and drop your lab report (PDF, JPG, PNG, or WebP) into the upload area.
4. **Process:** Click **Process Report with Gemini** to extract and view digitized results.
5. **Instant Demo:** Click **Load Sample Lab Report (Demo)** at any time to inspect the UI and grounding behaviors instantly.

---

## 🔒 Security & Medical Compliance Note

* **No Persistent Data Storage:** Reports and API keys are processed in memory during the request lifecycle and are not saved to disk or persistent databases.
* **Strict Anti-Fabrication:** MedLens enforces temperature `0.0` and structured Pydantic schemas to ensure deterministic, reproducible clinical data extraction.

---

## 👤 Author
* **GitHub:** [@Dheeraj2408](https://github.com/Dheeraj2408)

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
