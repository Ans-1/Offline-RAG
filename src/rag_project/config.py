import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Core Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- RAG Hyperparameters ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Mode Switching Logic ---
# Set RAG_MODE to "online" or "offline" in your .env file. Defaults to offline.
MODE = os.getenv("RAG_MODE", "offline").lower()

if MODE not in ["online", "offline"]:
    print(f"Warning: RAG_MODE '{MODE}' is invalid. Defaulting to 'offline'.")
    MODE = "offline"

if MODE == "online":
    # Online configuration (OpenAI)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        print("CRITICAL ERROR: OPENAI_API_KEY is missing in .env for online mode.")
        sys.exit(1)
        
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-4o-mini"

else:
    # Offline configuration (Local Models)
    OPENAI_API_KEY = None
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"       # Fast, local HuggingFace embeddings
    LLM_MODEL = "llama3"                       # Local Ollama model
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

print(f"⚙️ Config Loaded: System is running in [{MODE.upper()}] mode.")