📘🤖 Full-Context Question Paper Solver
No Chunking. No Missing Answers. Full Study Material.

A Streamlit-powered application that reads study material + question paper PDFs and generates complete, accurate answers using Google's Gemini 2.0 Flash model.

This version uses full-context input — no chunking, no truncation — ensuring the model sees everything it needs for perfect answers.

🚀 Features
✔ Full PDF → Text Extraction

Uses PyMuPDF to extract text cleanly from any PDF (BRPaper-proof).

✔ Full Study Material Context

Sends up to 300,000 characters directly to the LLM.
No chunking. No cutting. No “first 15k chars only” stupidity.

✔ AI Solves Every Question

Study material used as primary reference

If something is missing → model fills gaps using its own knowledge

Never outputs “Insufficient information”

✔ Accurate, Clean Answer Formatting

Each question is returned as:

Q1: <question>
A1: <answer>

✔ Complete Streamlit UI

Upload Study Material PDF

Upload Question Paper PDF

Preview extracted text

Generate answers

Download TXT file

🛠️ Installation
1️⃣ Clone the Repo
git clone https://github.com/yourusername/question-paper-solver.git
cd question-paper-solver

2️⃣ Create a Virtual Environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

🔑 Environment Variable Setup

Create a .env file:

GOOGLE_API_KEY=YOUR_GEMINI_API_KEY

▶️ Run the Application
streamlit run main_no_chunk.py


Your browser will open automatically:

http://localhost:8501

📂 Project Structure
📦 question-paper-solver
├── main_no_chunk.py      # Main Streamlit app
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .env                   # API key (not committed)

📦 requirements.txt Example

Use this if you haven’t created one yet:

streamlit
pymupdf
google-generativeai
python-dotenv

📸 Screenshots

(Optional — add these later)

Upload study material

Upload question paper

Preview extracted text

Full solved answers

🙌 Why This Project Exists

Most question-solver apps fail because they:

chunk text incorrectly

skip important study material

hallucinate

or say “insufficient information” all over the place

This app was built to fix all of that, using:

full-context prompts

strict formatting

better extraction

and smarter prompt engineering

🧠 Future Upgrades (Optional)

Advanced OCR (OpenCV preprocessing)

PDF → PDF solved output

Multi-PDF merging

Answer-source mapping

Offline LLM with CUDA (RTX 4060 support)

Just open an issue or request it.

🏆 License

MIT License — free to use, modify, and break however you like.
