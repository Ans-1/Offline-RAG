# 🧠 Local RAG Pipeline (Offline-First)

A modular, privacy-focused Retrieval-Augmented Generation (RAG) system built with modern Python tools. This project allows you to chat with your own PDF documents completely offline using local open-source models, ensuring your data never leaves your machine.

## ✨ Features

* **100% Offline Capability:** Powered by Ollama and HuggingFace for complete privacy.
* **Modern LangChain Architecture:** Utilizes the latest LangChain Expression Language (LCEL) for fast, reliable data routing.
* **Lightning-Fast Environment:** Managed by `uv`, ensuring reproducible builds in seconds without dependency hell.
* **Built-in Evaluation:** Includes an LLM-as-a-judge script to grade the RAG system's faithfulness and relevance against standard baselines.
* **Modular Design:** Easily swap between local models (Llama 3.2, Phi-3, Qwen) or cloud providers (OpenAI).

---

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed on your system:
1. **Git** (for version control)
2. **uv** (The blazingly fast Python package manager)
3. **Ollama** (For running local AI models)

---

## 🚀 Installation & Setup

**1. Clone the repository**
git clone https://github.com/Ans-1/Offline-RAG.git
cd rag-project

**2. Install dependencies via `uv`**
This single command reads the `uv.lock` file and perfectly recreates the environment.
uv sync

**3. Setup the Environment File**
Create a `.env` file in the root directory and set the mode to offline:
RAG_MODE=offline

**4. Pull the Local AI Model**
Make sure Ollama is running in the background, then pull a model like Llama 3.2:
ollama pull llama3.2

---

## 🏃‍♂️ Usage

**Step 0: Pre-Flight Check**
Ensure your environment is correctly configured before running the main scripts:
uv run python scripts/check_env.py

**Step 1: Add your Data**
Place any PDF files you want to chat with into the `data/raw/` directory.

**Step 2: Ingest the Documents**
This reads your PDFs, chunks the text, creates embeddings using Sentence-Transformers, and saves them to a local ChromaDB vector database.
uv run python -m src.rag_project.ingestion

**Step 3: Chat with your Data**
Start the interactive CLI to ask questions about your documents.
uv run python -m src.rag_project.retrieval

---

## 🧪 Testing & Utilities

This project includes several utility scripts in the `scripts/` folder for testing and evaluation:

* **Vanilla Chat (`scripts/chat.py`):** Talk directly to the LLM without RAG to test its baseline knowledge.
uv run python scripts/chat.py

* **RAG Battle (`scripts/compare.py`):** Pits the Baseline LLM against the RAG system and uses an "LLM-as-a-Judge" to score the RAG pipeline's faithfulness and relevance.
uv run python scripts/compare.py

---

## 📂 Project Structure

rag-project/
├── data/
│   └── raw/                 # Put your PDFs here
├── scripts/
│   ├── check_env.py         # Validates system readiness
│   ├── chat.py              # Direct model interaction
│   └── compare.py           # RAG Evaluation pipeline
├── src/
│   └── rag_project/
│       ├── __init__.py
│       ├── config.py        # Centralized settings and toggles
│       ├── ingestion.py     # Document loading & embedding logic
│       └── retrieval.py     # LCEL RAG chain and chat interface
├── vector_db/               # Auto-generated ChromaDB storage
├── .env                     # Local secrets (Ignored by Git)
├── pyproject.toml           # Project dependencies
└── uv.lock                  # Deterministic build lockfile

## 🤝 Troubleshooting

* **ModuleNotFoundError**: Ensure you are running scripts with `uv run python ...` so the virtual environment is activated.
* **Ollama Port Conflict**: If you see `bind: Only one usage of each socket address`, it means Ollama is already running in your system tray. You don't need to run `ollama serve` manually.
* **No context retrieved**: Ensure you have successfully run the ingestion script after placing PDFs in the `data/raw/` folder.