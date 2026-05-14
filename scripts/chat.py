import sys
from pathlib import Path

# Add the project root to the Python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.rag_project.retrieval import LLMFactory
from src.rag_project import config
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def run_vanilla_chat():
    print(f"\n{'='*50}")
    print(f"{'DIRECT MODEL CHAT':^50}")
    print(f"{'MODE: ' + config.MODE.upper() + ' | MODEL: ' + config.LLM_MODEL:^50}")
    print(f"{'='*50}")
    print("Type 'exit' to quit.\n")

    # Initialize the LLM using your existing factory
    llm = LLMFactory.get_llm()
    
    # Simple prompt template
    prompt = ChatPromptTemplate.from_template("{input}")
    chain = prompt | llm | StrOutputParser()

    while True:
        user_input = input("🧑 You: ")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("👋 Goodbye!")
            break
            
        if not user_input.strip():
            continue

        print("🤖 AI: ", end="", flush=True)
        
        try:
            # Stream the response for a better "chat" feel
            for chunk in chain.stream({"input": user_input}):
                print(chunk, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    run_vanilla_chat()