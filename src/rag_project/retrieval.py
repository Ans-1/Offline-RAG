import logging
from src.rag_project import config
from src.rag_project.ingestion import EmbeddingFactory

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Setup logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")

class LLMFactory:
    """Handles the creation of the LLM based on the system mode."""
    
    @staticmethod
    def get_llm():
        if config.MODE == "online":
            return ChatOpenAI(
                model=config.LLM_MODEL,
                api_key=config.OPENAI_API_KEY,
                temperature=0
            )
        else:
            return ChatOllama(
                model=config.LLM_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                temperature=0
            )

def format_docs(docs):
    """Utility function to format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain():
    # 1. Load Vector Database
    embedding_model = EmbeddingFactory.get_model()
    vectorstore = Chroma(
        persist_directory=str(config.VECTOR_DB_DIR),
        embedding_function=embedding_model
    )
    
    # 2. Create Retriever (fetches top 3 most relevant chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. Define the Prompt Template
    template = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, say that you don't know. 
    Use three sentences maximum and keep the answer concise.

    Context: {context}
    Question: {question}

    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # 4. Initialize LLM
    llm = LLMFactory.get_llm()
    
    # 5. Build the modern LCEL Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

def interactive_chat():
    print(f"\n🤖 RAG System Initialized [{config.MODE.upper()} MODE]")
    print("Type 'exit' or 'quit' to stop.\n")
    print("-" * 50)
    
    chain, retriever = build_rag_chain()
    
    while True:
        user_input = input("\n🧑 You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        print("🤖 AI: ", end="", flush=True)
        
        # Execute the chain
        response = chain.invoke(user_input)
        
        # Print the answer
        print(response)
        
        # Optional: Print sources for verification by running the retriever separately
        docs = retriever.invoke(user_input)
        print("\n[Sources]:")
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown").split("\\")[-1].split("/")[-1]
            page = doc.metadata.get("page", "Unknown")
            print(f"  - {source} (Page {page})")

if __name__ == "__main__":
    interactive_chat()