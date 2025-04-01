# vanillaRag

# RAG-Based Document Q&A with Groq LLM

This project is a **Streamlit-based application** that allows users to upload documents (PDF, DOCX, or TXT) and ask questions based on the content of the uploaded document. It uses **Groq LLM**, **DSPy**, **FAISS**, and **Sentence Transformers** to process the document and retrieve relevant answers.

---

## Features

- **Document Upload**: Supports PDF, DOCX, and TXT file formats.
- **Chunking and Embedding**: Splits the document into chunks and generates embeddings using `SentenceTransformer`.
- **FAISS Indexing**: Uses FAISS for efficient similarity search.
- **Question Answering**: Retrieves relevant chunks and generates answers using Groq LLM and DSPy.
- **Progress Bar**: Displays progress while processing large documents.

---

## Requirements

### Python Version
- Python 3.8 or higher

### Libraries
Install the required libraries using the following command:

```bash
pip install -r [requirements.txt](http://_vscodecontentref_/0)
