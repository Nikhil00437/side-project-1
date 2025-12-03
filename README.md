# 📘🤖 Full-Context Question Paper Solver

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent Streamlit application that reads entire study materials and question papers to generate complete, accurate answers using Google's Gemini 1.5 Flash model. No chunking, no context loss, just results.

This project was built to overcome the common failures of RAG (Retrieval-Augmented Generation) applications. By leveraging the massive context window of modern LLMs, we feed the entire study material directly to the model, eliminating errors caused by poor chunking, lost context, and insufficient information.

---

<!-- Optional: Add a GIF or screenshot of the app in action -->
<!-- ![App Demo](link-to-your-demo.gif) -->

## 🚀 Features

*   **✔️ Full PDF → Text Extraction:** Uses `PyMuPDF` to cleanly extract text from any PDF, ensuring high-fidelity input data for the language model.
*   **✔️ Massive Context Window:** Sends up to 300,000+ characters of study material directly to the LLM. No chunking, no truncation, no "first 15k chars only" limitations.
*   **✔️ AI Solves Every Question:**
    *   **Primary Source:** Uses the uploaded study material as the single source of truth.
    *   **Intelligent Gap Filling:** If the material is missing a minor detail, the model uses its internal knowledge to fill the gaps, but prioritizes the provided text.
    *   **No Dead Ends:** Engineered to always provide an answer and never output "Insufficient information."
*   **✔️ Accurate, Clean Formatting:** Each question and answer pair is returned in a strict, readable format for easy review.
    ```
    Q1: <question>
    A1: <answer>
    ```
*   **✔️ Complete Streamlit UI:** A simple and intuitive web interface to:
    *   Upload Study Material PDF
    *   Upload Question Paper PDF
    *   Preview extracted text from both documents
    *   Generate answers with a single click
    *   Download the complete solved paper as a `.txt` file

## 🛠️ Installation

Follow these steps to set up the project locally.

#### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/question-paper-solver.git
cd question-paper-solver
```

#### 2️⃣ Create and Activate a Virtual Environment

*   **Linux / macOS:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
*   **Windows:**
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```

#### 3️⃣ Install Dependencies
Install all required packages from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

#### 4️⃣ Environment Setup
Create a `.env` file in the root directory of the project and add your Google Gemini API key.
```
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
```

## ▶️ Usage

1.  Run the Streamlit application from your terminal:
    ```bash
    streamlit run main_no_chunk.py
    ```
2.  Your browser will automatically open the app at `http://localhost:8501`.

3.  **How to use the app:**
    *   Upload your **Study Material PDF** using the first file uploader.
    *   Upload your **Question Paper PDF** using the second file uploader.
    *   (Optional) Click the **Preview Extracted Text** tab to verify the text extraction.
    *   Click the **Generate Answers** button and wait for the AI to process.
    *   Once finished, a **Download Solved Paper** button will appear. Click it to save the answers as a `.txt` file.

## 📂 Project Structure

```
📦 question-paper-solver/
├── main_no_chunk.py      # Main Streamlit application
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── .env                  # API key (Local only - do not commit to Git)
```

**Requirements Reference:**
This project uses the following core libraries.
```
streamlit
pymupdf
google-generativeai
python-dotenv
```

## 🧠 Future Roadmap

This project is actively being developed. Future upgrades include:

- [ ] **Advanced OCR:** Integrate OpenCV for preprocessing scanned documents before OCR to improve text extraction from images and low-quality PDFs.
- [ ] **PDF → PDF Solved Output:** Generate a new PDF with the answers typed directly under each question.
- [ ] **Multi-PDF Merging:** Support for uploading and merging multiple study material PDFs into a single context.
- [ ] **Answer Source Mapping:** Add citations or references pointing to the page or section in the study material where the answer was found.
- [ ] **Offline LLM Support:** Implement support for locally-hosted LLMs with CUDA optimization (e.g., for RTX 40-series GPUs).

## 🙌 Contributing

Contributions are welcome! If you have an idea for an improvement or find a bug, please feel free to open an issue or submit a pull request.

## 🏆 License

This project is distributed under the MIT License. See the `LICENSE` file for more information. Free to use, modify, and break however you like.
