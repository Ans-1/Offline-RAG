import logging
from src.rag_project import config
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class EmbeddingFactory:
    """Handles the creation of embedding models based on the system mode."""
    
    @staticmethod
    def get_model():
        if config.MODE == "online":
            logging.info(f"Initializing Cloud Embeddings (OpenAI: {config.EMBEDDING_MODEL})...")
            return OpenAIEmbeddings(
                model=config.EMBEDDING_MODEL,
                api_key=config.OPENAI_API_KEY
            )
        else:
            logging.info(f"Initializing Local Embeddings (HuggingFace: {config.EMBEDDING_MODEL})...")
            return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

def run_ingestion():
    logging.info("Starting Data Ingestion Pipeline...")

    # 1. Identify files
    pdf_files = list(config.DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        logging.warning(f"No PDF files found in {config.DATA_DIR}. Please add some and try again.")
        return

    # 2. Load and Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, 
        chunk_overlap=config.CHUNK_OVERLAP
    )
    
    all_chunks = []
    for file_path in pdf_files:
        logging.info(f"Extracting text from: {file_path.name}")
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    logging.info(f"Split documents into {len(all_chunks)} total chunks.")

    # 3. Generate Embeddings and Store
    embedding_model = EmbeddingFactory.get_model()
    
    logging.info(f"Saving vectors to ChromaDB at {config.VECTOR_DB_DIR}...")
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_model,
        persist_directory=str(config.VECTOR_DB_DIR)
    )
    
    logging.info("✅ Ingestion completed successfully.")

if __name__ == "__main__":
    run_ingestion()