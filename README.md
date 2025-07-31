# RAG-UPDATED – Mini RAG CLI (Retrieval-Augmented Generation)

##  Model Download Required (Phi-2)

This project uses the **Phi-2 language model** in GGUF format for local answering via `llama-cpp-python`.  
Before running the script, you **must download the model file manually**:

### Step 1: Download the Phi-2 Model

Go to the Hugging Face page:
[https://huggingface.co/TheBloke/phi-2-GGUF](https://huggingface.co/TheBloke/phi-2-GGUF)

Choose a smaller quantized version (like Q4_0 or Q5_1) if you're on a lower-end machine.

### Step 2: Place the File

Put the downloaded `.gguf` file in your project root (same folder as `rag.py`).  
Or, if your script uses a subfolder, adjust the path like:
```python
model_path = "models/phi-2.Q4_K_M.gguf"
```

Recommended file:

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
