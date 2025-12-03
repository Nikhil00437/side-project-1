"""
Single-file Streamlit app:
- Upload study material (PDF or paste text)
- Upload question paper (PDF or paste questions)
- Answers each question using only the study material (RAG-style)
"""

import os
import streamlit as st
import pdfplumber
import google.generativeai as genai
import time
import numpy as np
from dotenv import load_dotenv
from typing import List, Tuple
from dotenv import load_dotenv
load_dotenv()


# ---------- CONFIG ----------
EMBEDDING_MODEL = "models/embedding-001"     # Gemini embeddings model
LLM_MODEL = "gemini-1.5-pro"                # answer generation model
CHUNK_SIZE = 2500                            # chars per chunk
TOP_K = 4                                    # top-k chunks to use per question
EMBED_BATCH = 16                             # batching for embeddings (if needed)
# ----------------------------

def configure_api():
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        st.error("GOOGLE_API_KEY or GEMINI_API_KEY not found. Set it in .env or environment variables.")
        st.stop()
    genai.configure(api_key=key)

def extract_text_from_pdf_filelike(file) -> str:
    """Extracts text from an uploaded PDF file-like object using pdfplumber."""
    text_parts = []
    try:
        with pdfplumber.open(file) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    text_parts.append(t)
    except Exception as e:
        st.error(f"PDF extraction error: {e}")
    return "\n\n".join(text_parts)

def naive_split_questions(text: str) -> List[str]:
    """
    Naive splitting of question paper into individual questions.
    Splits by common separators: numbered lines like '1.' 'Q1:' '1)' or blank lines.
    """
    # Try splitting on lines that look like question numbers
    lines = text.splitlines()
    qs = []
    curr = []
    for ln in lines:
        stripped = ln.strip()
        if stripped == "":
            # blank line could mean separation; if curr non-empty, commit
            if curr:
                qs.append(" ".join(curr).strip())
                curr = []
            continue
        # start of a numbered question
        if stripped[:3].lstrip().split(" ")[0].rstrip(").:").isdigit():
            # commit previous
            if curr:
                qs.append(" ".join(curr).strip())
            curr = [ln]
        elif stripped.lower().startswith("q") and (len(stripped) > 1 and stripped[1].isdigit()):
            if curr:
                qs.append(" ".join(curr).strip())
            curr = [ln]
        else:
            curr.append(ln)
    if curr:
        qs.append(" ".join(curr).strip())
    # final cleanup: filter out very short noise
    qs = [q for q in qs if len(q) > 10]
    return qs if qs else [text.strip()]  # fallback: whole text as one question

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        # try to break at newline for nicer chunks
        if end < n:
            tail = text[end:end+200]
            nl = tail.find("\n")
            if nl != -1:
                end = end + nl
        chunks.append(text[start:end].strip())
        start = end
    return chunks

# Embedding helpers
def get_embeddings(texts: List[str], model: str = EMBEDDING_MODEL, task_type: str = "retrieval_document") -> List[List[float]]:
    """Call Gemini embeddings API for a list of texts. Returns list of vectors."""
    embeddings = []
    # Process each text individually
    for i in range(0, len(texts)):
        result = genai.embed_content(
            model=model,
            content=texts[i],
            task_type=task_type
        )
        # Access embedding from result
        if hasattr(result, 'embedding'):
            vec = result.embedding
        else:
            vec = result['embedding']
        embeddings.append(vec)
        # tiny sleep to be polite in case of rate limiting
        time.sleep(0.05)
    return embeddings

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def retrieve_top_k(question: str, chunk_texts: List[str], chunk_vecs: List[List[float]], k: int = TOP_K) -> List[Tuple[str, float]]:
    q_vec = get_embeddings([question], task_type="retrieval_query")[0]
    qv = np.array(q_vec)
    scores = []
    for txt, vec in zip(chunk_texts, chunk_vecs):
        sc = cosine_sim(qv, np.array(vec))
        scores.append((txt, sc))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]

def build_answer_prompt(question: str, retrieved_chunks: List[Tuple[str, float]]) -> str:
    """
    Build a strict prompt that instructs the model to answer only from the provided context.
    """
    context = "\n\n---\n\n".join([f"Chunk (score {score:.4f}):\n{chunk}" for chunk, score in retrieved_chunks])
    prompt = f"""
You are an expert assistant. You MUST answer the question using ONLY the STUDY MATERIAL provided below.
Do NOT use outside information. If the STUDY MATERIAL does NOT contain enough information to answer,
respond exactly with: "Insufficient material to answer this question."

STUDY MATERIAL (relevant chunks):
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Provide a concise answer. If you can, include 1-2 sentence justification referencing which chunk (by quoting a short fragment of the chunk) you used.
- If the material doesn't contain the information, say "Insufficient material to answer this question."
- Keep answers factual and avoid hallucination.
"""
    return prompt

def call_llm_for_answer(prompt: str, model: str = LLM_MODEL, temperature: float = 0.0, max_tokens: int = 800) -> str:
    # Combine system and user messages for Gemini
    full_prompt = f"You are a helpful assistant that answers questions using only the provided study material.\n\n{prompt}"
    
    # Create the model instance
    gemini_model = genai.GenerativeModel(model)
    
    # Generate content with temperature
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens
    )
    
    response = gemini_model.generate_content(
        full_prompt,
        generation_config=generation_config
    )
    
    return response.text.strip()

# Streamlit UI
def main():
    st.set_page_config(page_title="Study PDF → QPaper PDF → Answers", layout="wide")
    st.title("Study material → Question paper → Answers (RAG, single-file)")

    configure_api_help = st.sidebar.checkbox("Configure API (show instructions)", value=False)
    if configure_api_help:
        st.sidebar.markdown(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY in a `.env` file or system environment variables.\n\n"
            "Example `.env`:\n```\nGOOGLE_API_KEY=your-api-key-here\n```"
        )

    configure_api()  # ensures API key exists

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1) Study material (source)")
        study_mode = st.radio("Study input mode", ["Paste / Type text", "Upload PDF"], index=1)
        study_text = ""
        if study_mode == "Paste / Type text":
            study_text = st.text_area("Paste or type the study material here", height=250)
        else:
            study_pdf = st.file_uploader("Upload study-material PDF", type=["pdf"], key="study_pdf")
            if study_pdf is not None:
                with st.spinner("Extracting study PDF..."):
                    study_text = extract_text_from_pdf_filelike(study_pdf)
                    st.success("Extracted study material (preview shown)")
        if study_text:
            st.markdown("**Preview (first 3000 chars)**")
            st.code(study_text[:3000] + ("..." if len(study_text) > 3000 else ""), language="text")

    with col2:
        st.subheader("2) Question paper (input)")
        q_mode = st.radio("Question input mode", ["Upload PDF", "Paste / Type questions"], index=0)
        questions_text = ""
        if q_mode == "Upload PDF":
            q_pdf = st.file_uploader("Upload question-paper PDF", type=["pdf"], key="qpdf")
            if q_pdf is not None:
                with st.spinner("Extracting question PDF..."):
                    questions_text = extract_text_from_pdf_filelike(q_pdf)
                    st.success("Extracted questions (preview shown)")
        else:
            questions_text = st.text_area("Paste/Type the question paper text here", height=250)
        if questions_text:
            st.markdown("**Question preview (first 2000 chars)**")
            st.code(questions_text[:2000] + ("..." if len(questions_text) > 2000 else ""), language="text")

    st.markdown("---")
    st.subheader("Settings")
    top_k = st.number_input("Top-k retrieved chunks per question", min_value=1, max_value=10, value=TOP_K)
    chunk_chars = st.number_input("Chunk size (characters)", min_value=500, max_value=8000, value=CHUNK_SIZE, step=100)
    model = st.text_input("LLM model", value=LLM_MODEL)
    emb_model = st.text_input("Embedding model", value=EMBEDDING_MODEL)
    temp = st.slider("Temperature (creativity)", 0.0, 1.0, 0.0)
    run_btn = st.button("Generate Answers")

    if run_btn:
        if not study_text or not study_text.strip():
            st.error("No study material provided.")
            st.stop()
        if not questions_text or not questions_text.strip():
            st.error("No question paper provided.")
            st.stop()

        with st.spinner("Preparing study chunks..."):
            chunks = chunk_text(study_text, chunk_size=chunk_chars)
            st.write(f"Study material split into {len(chunks)} chunk(s). Sending for embeddings...")

        # compute embeddings for chunks
        with st.spinner("Computing embeddings for study chunks..."):
            chunk_vecs = get_embeddings(chunks, model=emb_model)
            st.success("Embeddings ready.")

        # parse questions
        with st.spinner("Parsing questions..."):
            questions = naive_split_questions(questions_text)
            st.write(f"Identified {len(questions)} question(s).")

        answers = []
        # process each question
        for idx, q in enumerate(questions, start=1):
            st.info(f"Processing Question {idx}/{len(questions)}")
            # retrieve
            retrieved = retrieve_top_k(q, chunks, chunk_vecs, k=top_k)
            # build prompt and call LLM
            prompt = build_answer_prompt(q, retrieved)
            try:
                ans = call_llm_for_answer(prompt, model=model, temperature=temp)
            except Exception as e:
                ans = f"[LLM error: {e}]"
            answers.append((q, ans))

            # small throttle
            time.sleep(0.2)

        # Display results in two columns
        st.markdown("### Results")
        for i, (q, a) in enumerate(answers, start=1):
            st.markdown(f"**Q{i}.** {q}")
            st.markdown(f"**Answer:**\n\n{a}")
            st.markdown("---")

        # Download as TXT
        combined = []
        for i, (q, a) in enumerate(answers, start=1):
            combined.append(f"Q{i}. {q}\nAnswer:\n{a}\n\n")
        output_txt = "\n".join(combined)
        st.download_button("Download answers (TXT)", data=output_txt, file_name="answers.txt", mime="text/plain")
        st.success("Done. Review the answers above.")

if __name__ == "__main__":
    main()
