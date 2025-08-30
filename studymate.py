# studymate_app.py

import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
qa_model = pipeline("text2text-generation", model="google/flan-t5-base")  # Replace with IBM Watsonx later

# Helper: Extract text from PDF
def extract_text(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Helper: Chunk text
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Helper: Embed and index chunks
def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings, chunks

# Helper: Retrieve relevant chunks
def search_chunks(query, index, chunks, embeddings, top_k=3):
    query_vec = embedder.encode([query])
    query_vec = np.array(query_vec).astype('float32')
    _, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0]]

# Helper: Generate answer
def generate_answer(context, question):
    prompt = f"Answer the question based on the context:\nContext: {context}\nQuestion: {question}"
    result = qa_model(prompt, max_length=200)[0]['generated_text']
    return result

# Streamlit UI
st.title("📚 StudyMate: AI-Powered PDF Q&A")

uploaded_file = st.file_uploader("Upload your academic PDF", type="pdf")
question = st.text_input("Ask a question from your study material")

if uploaded_file and question:
    try:
        with st.spinner("Processing PDF..."):
            pdf_bytes = uploaded_file.read()
            text = extract_text(pdf_bytes)
            if not text.strip():
                st.error("No text found in the PDF.")
            else: 
                chunks = chunk_text(text)
                index, embeddings, chunks = create_faiss_index(chunks)
                relevant_chunks = search_chunks(question, index, chunks, embeddings)
                context = " ".join(relevant_chunks)
                answer = generate_answer(context, question)

        st.subheader("📖 Answer")
        st.write(answer)

        with st.expander("🔍 Context Used"):
            st.write(context)
    except Exception as e:
        st.error(f"An error occurred: {e}")