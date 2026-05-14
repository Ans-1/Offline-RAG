import sys
from pathlib import Path

# Add the project root (one level up from this script) to the Python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# NOW you can import your project files
from src.rag_project import config
from src.rag_project.retrieval import LLMFactory, build_rag_chain, format_docs
import sys
from src.rag_project import config
from src.rag_project.retrieval import LLMFactory, build_rag_chain, format_docs
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_direct_response(question):
    """Queries the LLM directly without any document context."""
    llm = LLMFactory.get_llm()
    prompt = ChatPromptTemplate.from_template("{question}")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})

def evaluate_rag(question, context, answer):
    """Uses the LLM as a judge to score the RAG response."""
    llm = LLMFactory.get_llm()
    
    judge_template = """You are an expert auditor. Evaluate the RAG response based on the context.
    
    [QUESTION]: {question}
    [CONTEXT]: {context}
    [ANSWER]: {answer}
    
    Provide your evaluation in exactly this format:
    FAITHFULNESS: [Score 1-5]/5 - Brief justification.
    RELEVANCE: [Score 1-5]/5 - Brief justification.
    """
    prompt = ChatPromptTemplate.from_template(judge_template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": context, "answer": answer})

def run_battle(questions):
    print(f"\n{'='*80}")
    print(f"{'RAG BATTLE: DOCUMENT VS. GENERAL KNOWLEDGE':^80}")
    print(f"{'MODE: ' + config.MODE.upper():^80}")
    print(f"{'='*80}\n")
    
    rag_chain, retriever = build_rag_chain()
    
    for i, q in enumerate(questions, 1):
        print(f"ROUND {i}: {q}")
        print("-" * 80)
        
        # --- 1. BASELINE ---
        print("\n[ ❌ BASELINE: WITHOUT RAG ]")
        direct_res = get_direct_response(q)
        print(f"{direct_res.strip()}")
        print("." * 40)
            
        # --- 2. RAG ---
        print("\n[ ✅ SYSTEM: WITH RAG ]")
        rag_res = rag_chain.invoke(q)
        print(f"{rag_res.strip()}")
        
        # --- 3. SOURCES ---
        docs = retriever.invoke(q)
        source_names = list(set([doc.metadata.get("source", "Unknown").split("/")[-1] for doc in docs]))
        print(f"\n[ 📑 DATA SOURCES: {', '.join(source_names)} ]")
        print("." * 40)

        # --- 4. EVALUATION ---
        print("\n[ ⚖️ JUDGE'S EVALUATION ]")
        context_str = format_docs(docs)
        if context_str.strip():
            eval_score = evaluate_rag(q, context_str, rag_res)
            print(eval_score.strip())
        else:
            print("   (Evaluation skipped: No context retrieved)")
        
        print(f"\n{'='*80}")

if __name__ == "__main__":
    test_questions = [
        "What is the specific temperature requirement for the instruments?",
        "Who is the lead author of this document?",
        "What does the document say about the use of AI in space?"
    ]
    run_battle(test_questions)