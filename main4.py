import os
import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
import time
from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIG ----------------
DEFAULT_MODEL = "gemini-2.0-flash"
MAX_CONTEXT_CHARS = 300000   # full context allowed safely
MAX_RETRY = 3
# ----------------------------------------


# Configure Gemini API
def configure_api():
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        st.error("API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY in environment variables.")
        st.stop()
    genai.configure(api_key=key)
    return key


# PDF Extraction (clean, direct, no stupid chunking)
def extract_text_from_pdf(uploaded_file):
    try:
        raw_bytes = uploaded_file.read()
        doc = fitz.open(stream=raw_bytes, filetype="pdf")

        full_text = ""
        for page in doc:
            page_text = page.get_text("text") or ""
            full_text += page_text + "\n\n"

        # clean watermark trash if any
        cleaned = full_text.replace("brpaper.com", "").strip()
        return cleaned

    except Exception as e:
        return f"[PDF extraction error: {e}]"


# Build prompt — CLEAN VERSION
def build_prompt(study_material: str, question_paper: str) -> str:
    return f"""
You are an expert exam solver with deep domain knowledge.

Use the study material FIRST.  
If something is missing, answer using your own knowledge so the answer is complete.  
Never say "insufficient information".  
Just answer fully.

====================
STUDY MATERIAL:
====================
{study_material}

====================
QUESTION PAPER:
====================
{question_paper}

====================
TASK:
====================
For every question in the question paper:

Q<n>: <exact question>  
A<n>: <full answer>

Do not skip any question.  
Do not write anything extra.  
Just give perfect answers.
"""


# Gemini Call
def call_gemini(prompt: str, model: str, temperature: float):
    last_exc = None

    for attempt in range(MAX_RETRY):
        try:
            gm = genai.GenerativeModel(model)

            config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=8000
            )

            response = gm.generate_content(prompt, generation_config=config)
            return response.text

        except Exception as e:
            last_exc = e
            time.sleep(1 + attempt * 2)

    raise last_exc


# ---------------- UI ----------------
def main():
    st.set_page_config(page_title="Full Context Question Solver", layout="wide")
    st.title("📘🤖 Full-Context Question Paper Solver (NO CHUNKING)")

    st.markdown("""
    Upload:
    - **Study Material PDF**
    - **Question Paper PDF**

    AI will solve EVERYTHING using full text.  
    No chunking. No cutting. No excuses.
    """)

    configure_api()

    # Upload fields
    st.subheader("Upload Study Material PDF")
    study_pdf = st.file_uploader("Select study material", type=["pdf"])

    st.subheader("Upload Question Paper PDF")
    question_pdf = st.file_uploader("Select question paper", type=["pdf"])

    study_material = ""
    questions_text = ""

    # Extract study material
    if study_pdf:
        with st.spinner("Extracting study material..."):
            study_material = extract_text_from_pdf(study_pdf)
            st.success("Study material extracted.")
        st.code(study_material[:5000] + "\n...[truncated preview]...", language="text")

    # Extract questions
    if question_pdf:
        with st.spinner("Extracting question paper..."):
            questions_text = extract_text_from_pdf(question_pdf)
            st.success("Question paper extracted.")
        st.code(questions_text[:5000] + "\n...[truncated preview]...", language="text")

    st.markdown("---")

    # Settings
    model = st.text_input("Model", DEFAULT_MODEL)
    temp = st.slider("Creativity", 0.0, 1.0, 0.1)

    run_btn = st.button("Solve Question Paper")

    if run_btn:
        if not study_material or not questions_text:
            st.error("Upload BOTH PDFs before solving.")
            st.stop()

        # full context, trimmed to safety limits
        study_context = study_material[:MAX_CONTEXT_CHARS]

        prompt = build_prompt(study_context, questions_text)

        # solve
        with st.spinner("Solving questions... hold on, genius at work."):
            try:
                answers = call_gemini(prompt, model=model, temperature=temp)
            except Exception as e:
                st.error(f"Gemini Error: {e}")
                st.stop()

        st.subheader("✅ Solved Answers")
        st.text_area("Answers", answers, height=600)

        st.download_button(
            "Download TXT",
            data=answers,
            file_name="solved_question_paper.txt",
            mime="text/plain"
        )

        st.success("Done. Enjoy your fully solved paper.")


if __name__ == "__main__":
    main()
