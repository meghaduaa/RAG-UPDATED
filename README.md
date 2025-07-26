# RAG-UPDATED
# Mini RAG CLI (Retrieval-Augmented Generation)

This is a lightweight, fast, and local Retrieval-Augmented Generation (RAG) system built using [LangChain](https://python.langchain.com), FAISS, and a tiny local LLM (`Phi-2` via `LlamaCpp`). Perfect for running document-based question-answering tasks right from your terminal.

---

## Features

- Loads all `.pdf`, `.docx`, and `.txt` documents from the `document/` folder  
- Splits them into manageable chunks  
- Embeds them using `sentence-transformers/all-MiniLM-L6-v2`  
- Stores in a local FAISS vector store  
- Uses `Phi-2` locally via `llama-cpp-python` for answering queries  
- Works fully offline (after initial model download)  
- CLI-based, minimal setup, no Hugging Face login required  

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/mini-rag-cli.git
cd mini-rag-cli
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` yet, create one with the following contents:

```txt
langchain
langchain-community
langchain-core
langchain-huggingface
transformers
torch
faiss-cpu
sentence-transformers
python-docx
pypdf
docx2txt
```

### 3. Add your documents

Place your `.pdf`, `.docx`, and `.txt` files inside a folder named `document/` in the same directory as `rag.py`.

---

## Run the QA CLI

```bash
python rag.py
```

You’ll see output like:

```
Loading and processing documents...
Embedding and storing in FAISS...
Loading Phi-2 model...
Creating QA chain...

Ask a question (type 'exit' to quit):
>>>
```

You can now ask questions based on the content in your documents.

---

## Project Structure

```
mini-rag-cli/
├── rag.py               # Main CLI script
├── document/            # Your input PDF, DOCX, and TXT files
└── README.md            # Project documentation
```

---

## Model Details

- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2`  
- **LLM for generation**: `Phi-2` (GGUF model) via `llama-cpp-python`  

> Ensure you have the `phi-2.Q4_K_M.gguf` model file in your project directory.

You can download Phi-2 from [TheBloke’s Hugging Face page](https://huggingface.co/TheBloke/phi-2-GGUF).

---

## Contributing

PRs and issues are welcome!  
Feel free to contribute better prompting strategies, faster LLMs, or UI improvements.

---
