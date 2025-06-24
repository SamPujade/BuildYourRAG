# BuildYourRAG  
*A modular RAG (Retrieval-Augmented Generation) framework to chat with your own documents.*

## Overview

**BuildYourRAG** is a Python-based application that empowers users to build custom RAG pipelines with ease. Featuring a **FastAPI** backend and **Streamlit** frontend, this app allows users to upload documents and interact with them via natural language queries. Embeddings are managed using **ChromaDB**, and the LLM support is currently API-based with **Gemini**.

---

## Features

- Upload and chat with your documents  
- API-based LLM integration (Gemini)  
- ChromaDB-powered vector store  
- Streamlit + FastAPI architecture  
- Easy dependency management with `uv` and `pyproject.toml`  

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/SamPujade/BuildYourRAG.git
cd BuildYourRAG
```

### 2. Setup environment variables

Create a `.env` file in the root of your project with the following content:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 2. Install dependencies and start the app

Run the following script to install dependencies and launch both frontend and backend:

```bash
bash start
```

- Backend runs at: `http://localhost:8199`
- Frontend runs in your browser via Streamlit

Logs are saved to `logs/backend.log` and `logs/frontend.log`.

## Project Structure

```
.
├── start.sh                 # Script to start backend and frontend
├── pyproject.toml           # Project dependencies managed by uv
├── .env                     # API keys and secrets (user-provided)
├── src/
│   ├── app.py               # Streamlit frontend app
│   └── server/
│       └── main.py          # FastAPI backend entry point
├── logs/                    # Log files for debugging
└── data/                    # Data folder
    └── chroma_data/         # ChromaDB collection folder
        └── upload/          # Uploaded folders

```

## 📌 Requirements

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) (modern Python package manager)

Install `uv` if not already installed: https://docs.astral.sh/uv/getting-started/installation/