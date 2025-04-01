
import streamlit as st
import os
import faiss
import numpy as np
import dspy
from PyPDF2 import PdfReader
from groq import Groq
from sentence_transformers import SentenceTransformer
from docx import Document
from more_itertools import chunked  # For batching embeddings

#Use Groq API directly
GROQ_API_KEY = ""
groq_client = Groq(api_key=GROQ_API_KEY)

# Correct DSPy Model Initialization
if "dspy_configured" not in st.session_state:
    llm = dspy.LM(provider="groq", model="groq/qwen-qwq-32b", api_key=GROQ_API_KEY)
    dspy.configure(lm=llm)
    st.session_state["dspy_configured"] = True

#  Initialize Embedding Model
embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")  # Faster model

#  Initialize FAISS index
index = faiss.IndexFlatL2(384)  # 384 is the embedding size for MiniLM
text_chunks = []

#  Process Document Uploads
def process_document(file):
    global index, text_chunks
    text_chunks.clear()

    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        text = file.getvalue().decode("utf-8")

    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]  # Larger chunk size
    text_chunks.extend(chunks)

    index.reset()
    progress_bar = st.progress(0)  # Initialize progress bar
    total_batches = len(chunks) // 32 + 1

    for i, batch in enumerate(chunked(chunks, 32)):  # Process 32 chunks at a time
        embeddings = embedding_model.encode(batch)
        index.add(np.array(embeddings, dtype=np.float32))
        progress_bar.progress((i + 1) / total_batches)  # Update progress bar

    st.session_state["processed"] = True
    st.success(" Document processed successfully!")

#  Retrieve Relevant Chunks
def retrieve_relevant_chunks(query):
    if index.ntotal == 0:  # Check if the FAISS index is empty
        st.warning(" No embeddings found. Please upload and process a document first.")
        return []

    query_embedding = embedding_model.encode([query])
    _, indices = index.search(np.array(query_embedding, dtype=np.float32), k=3)
    return [text_chunks[i] for i in indices[0] if i < len(text_chunks)]

#  DSPy RAG Module
class RAGAnswerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("context, question -> answer")

    def forward(self, context, question):
        return self.predict(context=context, question=question)

rag_model = RAGAnswerModule()

#  Generate Answer with DSPy + Groq
def generate_answer(query):
    relevant_text = retrieve_relevant_chunks(query)
    context = "\n".join(relevant_text)
    
    response = rag_model.forward(context=context, question=query)
   
    return response.answer.strip()

#  Streamlit UI
st.title("RAG-Based Document Q&A with Groq LLM")

uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT document", type=["pdf", "docx", "txt"])
if uploaded_file:
    process_document(uploaded_file)

query = st.text_input("Ask a question based on the document")
if query:
    if "processed" in st.session_state:
        answer = generate_answer(query)
        st.subheader("Answer:")
        st.write(answer)
    else:
        st.warning(" Please upload and process a document first.")