# Legacy AI — RAG assistant for legacy codebases

Legacy AI is a small tool that helps you **understand and document legacy projects** by using a **RAG (Retrieval-Augmented Generation)** pipeline over a codebase.
It ingests a repository, chunks files, stores embeddings in a vector database, and lets you ask questions through a simple UI.

## Features
- Ingest a codebase (multi-file, multi-language)
- Chunking adapted for code/text
- Vector store with **ChromaDB**
- RAG pipeline with **LangChain**
- Local LLM inference with **Ollama**
- Simple UI with **Gradio**
- Q&A over the repository: architecture, file responsibilities, functions/classes, configuration, etc.

## Tech stack
- **Python**
- **LangChain**
- **ChromaDB**
- **Ollama**
- **Gradio**

## How it works (high level)
1. **Ingestion**: scan the repository and load supported files  
2. **Chunking**: split files into retrievable chunks (code-aware splitting when possible)  
3. **Embedding + storage**: generate embeddings and store them in **ChromaDB**  
4. **Retrieval**: fetch the most relevant chunks for a user query  
5. **Generation**: send retrieved context + question to the LLM (Ollama)  
6. **UI**: ask questions from a Gradio interface

## Requirements
- Python 3.10+ (recommended)
- Ollama installed and running
- An Ollama model pulled locally (example below)

## Installation
```bash
git clone <YOUR_REPO_URL>
cd legacy_ai
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
