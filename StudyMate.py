# app.py
import streamlit as st
import fitz
import faiss
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import os

# -----------------------------
# Config / Page
# -----------------------------
st.set_page_config(page_title="StudyMate (Granite)", layout="wide")
st.markdown("<h1 style='text-align:center;color:#6A5ACD;'>StudyMate — PDF Q&A (Granite)</h1>",
            unsafe_allow_html=True)
st.sidebar.title("Tasks")
task = st.sidebar.radio("Choose", ["Upload PDFs", "Generate Questions (Granite)", "Answer Questions", "Ask Question"])

# -----------------------------
# Fast models loaded at start
# -----------------------------
@st.cache_resource
def load_fast_models():
    # extractive QA (fast)
    qa_pipe = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    # sentence embeddings
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return qa_pipe, embed_model

qa_pipeline, embedder = load_fast_models()

# -----------------------------
# Granite: lazy loaded on demand
# -----------------------------
@st.cache_resource
def load_granite():
    """
    Load Granite tokenizer + model once (lazy). This model ~2B params -> may still be large.
    Ensure you have enough RAM; loads on CPU if no GPU.
    """
    model_id = "ibm-granite/granite-3.2-2b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    # move to device if gpu available
    device = 0 if torch.cuda.is_available() else -1
    if device >= 0:
        model = model.to("cuda")
    return tokenizer, model

def generate_with_granite(prompt, max_new_tokens=200):
    """
    Generate text using granite. We use tokenizer.apply_chat_template if available (per your snippet),
    otherwise use a simple prompt -> generate flow.
    """
    tokenizer, model = load_granite()
    model_device = next(model.parameters()).device

    # Try preferred chat template API if available
    try:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        # move tensors to model device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        # generated tokens after input length
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    except Exception:
        # fallback simple encode->generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model_device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated.strip()

# -----------------------------
# IBM Watson TTS helper (optional)
# -----------------------------
def get_ibm_tts():
    api_key = os.getenv("IBM_WATSON_APIKEY", "")
    url = os.getenv("IBM_WATSON_URL", "")
    if not api_key or not url:
        return None
    auth = IAMAuthenticator(api_key)
    tts = TextToSpeechV1(authenticator=auth)
    tts.set_service_url(url)
    return tts

# -----------------------------
# PDF helpers
# -----------------------------
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    txt = []
    for i, page in enumerate(doc):
        page_text = page.get_text().strip()
        if page_text:
            # include page number metadata for later reference
            txt.append((i + 1, page_text))
    # return combined text but keep page splits for chunking reference if needed
    return txt

def chunk_text_from_pages(page_tuples, chunk_size_words=400, overlap=50):
    """
    Input: list of tuples (page_no, page_text)
    Output: list of dicts: {"page": int, "text": str}
    """
    chunks = []
    for page_no, page_text in page_tuples:
        words = page_text.split()
        step = chunk_size_words - overlap
        if step <= 0:
            step = chunk_size_words
        for i in range(0, max(1, len(words)), step):
            chunk_words = words[i:i + chunk_size_words]
            if chunk_words:
                chunks.append({"page": page_no, "text": " ".join(chunk_words)})
    return chunks

# -----------------------------
# Embedding + FAISS helpers
# -----------------------------
def create_faiss_index(text_chunks):
    # text_chunks: list of dicts with "text"
    texts = [c["text"] for c in text_chunks]
    embs = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    embs = np.array(embs).astype("float32")
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
    return index, embs

def retrieve_top_k(query, index, text_chunks, k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    k = min(k, index.ntotal) if index.ntotal > 0 else 0
    if k == 0:
        return []
    D, I = index.search(q_emb, k)
    results = []
    for idx in I[0]:
        chunk = text_chunks[idx]
        results.append(chunk)
    return results

# -----------------------------
# Session variables
# -----------------------------
if "pages" not in st.session_state:
    st.session_state.pages = []             # list of (page_no, text)
if "chunks" not in st.session_state:
    st.session_state.chunks = []            # list of {"page","text"}
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "granite_loaded" not in st.session_state:
    st.session_state.granite_loaded = False

# -----------------------------
# UI: Upload PDFs
# -----------------------------
if task == "Upload PDFs":
    st.header("Upload one or more PDFs")
    uploaded = st.file_uploader("Academic PDFs", type="pdf", accept_multiple_files=True)
    if uploaded:
        with st.spinner("Extracting text & chunking..."):
            pages_all = []
            for f in uploaded:
                pages = extract_text_from_pdf(f)
                pages_all.extend(pages)
            st.session_state.pages = pages_all
            st.session_state.chunks = chunk_text_from_pages(pages_all, chunk_size_words=500, overlap=80)
            if st.session_state.chunks:
                idx, embs = create_faiss_index(st.session_state.chunks)
                st.session_state.faiss_index = idx
                st.success(f"Processed {len(uploaded)} file(s): {len(st.session_state.chunks)} chunks created.")
            else:
                st.error("No text extracted from PDFs.")

# -----------------------------
# UI: Generate Questions (Granite)
# -----------------------------
elif task == "Generate Questions (Granite)":
    st.header("Auto-generate questions using IBM Granite")
    st.write("Granite is loaded only when you click Generate (avoids long startup).")

    if not st.session_state.chunks:
        st.warning("Upload PDFs first under 'Upload PDFs' tab.")
    else:
        num_chunks_to_use = st.slider("Use how many chunks to source questions from (top chunks)", 1, min(10, len(st.session_state.chunks)), min(5, len(st.session_state.chunks)))
        if st.button("Generate Questions (Granite)"):
            with st.spinner("Loading Granite (may take a few seconds) and generating..."):
                # mark granite loaded
                st.session_state.granite_loaded = True
                # build context from first N chunks
                ctx_texts = [c["text"] for c in st.session_state.chunks[:num_chunks_to_use]]
                context = "\n\n".join(f"- {t}" for t in ctx_texts)
                prompt = f"From the context below, generate 8 clear exam-style questions (short) that only use the provided context.\n\nContext:\n{context}\n\nQuestions:"
                try:
                    generated = generate_with_granite(prompt, max_new_tokens=256)
                except Exception as e:
                    st.error(f"Granite generation error: {e}")
                    generated = ""
                # split and keep non-empty lines
                questions = [q.strip() for q in (generated.split("\n") if generated else []) if q.strip()]
                if not questions:
                    st.warning("No questions generated. Try increasing the chunk count or reupload PDFs.")
                else:
                    st.session_state.qa_history.append({"type": "generated_questions", "payload": questions})
                    st.success(f"Generated {len(questions)} questions.")
                    for q in questions:
                        st.write("•", q)

# -----------------------------
# UI: Answer Questions (use generated)
# -----------------------------
elif task == "Answer Questions":
    st.header("Answer auto-generated questions (extractive QA)")
    # find generated questions in history
    gen_qs = []
    for entry in st.session_state.qa_history:
        if entry.get("type") == "generated_questions":
            gen_qs = entry.get("payload", [])
            break

    if not gen_qs:
        st.warning("No generated questions in this session. Run 'Generate Questions (Granite)' first.")
    else:
        for q in gen_qs:
            with st.expander(q, expanded=False):
                if st.button(f"Answer: {q}", key=f"ans_{q[:40]}"):
                    # retrieve context
                    retrieved = retrieve_top_k(q, st.session_state.faiss_index, st.session_state.chunks, k=3)
                    ctx = " ".join([r["text"] for r in retrieved])
                    if not ctx.strip():
                        st.info("No context available to answer this question.")
                        continue
                    ans = qa_pipeline(question=q, context=ctx)
                    answer_text = ans.get("answer") if isinstance(ans, dict) else str(ans)
                    st.markdown(f"**Answer:** {answer_text}")
                    st.markdown("**Referenced pages:** " + ", ".join(str(r["page"]) for r in retrieved))

# -----------------------------
# UI: Ask a custom question
# -----------------------------
elif task == "Ask Question":
    st.header("Ask any question from the uploaded PDFs")
    user_q = st.text_input("Enter your question here:")
    if st.button("Get Answer") and user_q.strip():
        if not st.session_state.faiss_index:
            st.warning("Please upload PDFs first.")
        else:
            with st.spinner("Retrieving context and answering..."):
                retrieved = retrieve_top_k(user_q, st.session_state.faiss_index, st.session_state.chunks, k=4)
                ctx = " ".join(r["text"] for r in retrieved)
                if not ctx.strip():
                    st.info("No relevant text found in documents.")
                else:
                    ans = qa_pipeline(question=user_q, context=ctx)
                    answer_text = ans.get("answer") if isinstance(ans, dict) else str(ans)
                    st.success(answer_text)
                    st.markdown("**Referenced pages:** " + ", ".join(str(r["page"]) for r in retrieved))

                    # Save to history
                    st.session_state.qa_history.append({"type":"qa", "question":user_q, "answer":answer_text, "chunks":[r for r in retrieved]})

    # Optional TTS
    if st.button("🔊 Speak last answer"):
        tts = get_ibm_tts()
        if tts is None:
            st.error("IBM Watson TTS credentials not set in env (IBM_WATSON_APIKEY, IBM_WATSON_URL).")
        else:
            if not st.session_state.qa_history:
                st.info("No answer yet to speak.")
            else:
                last = st.session_state.qa_history[-1]
                text_to_speak = last.get("answer", "")
                if not text_to_speak:
                    st.info("Last entry has no answer to speak.")
                else:
                    audio = tts.synthesize(text_to_speak, voice="en-US_AllisonV3Voice", accept="audio/mp3").get_result().content
                    st.audio(audio, format="audio/mp3")

# -----------------------------
# Sidebar: Show History & Download
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Session History")
for i, e in enumerate(reversed(st.session_state.qa_history[-10:]), 1):
    if e.get("type") == "generated_questions":
        st.sidebar.write(f"Gen Qs: {len(e.get('payload',[]))}")
    else:
        st.sidebar.write(f"Q: {e.get('question','')}")

def download_transcript():
    lines = []
    for e in st.session_state.qa_history:
        if e.get("type")=="generated_questions":
            lines.append("Generated questions:")
            for q in e.get("payload",[]):
                lines.append("Q: " + q)
            lines.append("---")
        else:
            lines.append("Q: " + e.get("question",""))
            lines.append("A: " + e.get("answer",""))
            lines.append("---")
    return "\n".join(lines)

st.sidebar.download_button("Download Session", download_transcript(), file_name="study_session.txt", mime="text/plain")