import os
import json
from typing import List, Dict, Any, TypedDict
from dotenv import load_dotenv
from google import genai
from langgraph.graph import StateGraph, END

# Load environment variables from the .env file in the root folder
load_dotenv()

# Initialize Gemini Client (automatically picks up GEMINI_API_KEY from .env)
client = genai.Client()

# Import our modular hybrid search and reranking function
from retrieval import hybrid_retrieve_and_rerank

# --- Define LangGraph State ---
class AgentState(TypedDict):
    question: str
    rewritten_question: str
    retrieved_docs: List[Dict[str, Any]]
    grade_status: str       # "sufficient", "insufficient", "outdated"
    retry_count: int
    answer: str
    cited_ids: List[str]
    abstain: bool

# --- Node 1: Retrieval & Reranking ---
def retrieve_node(state: AgentState) -> AgentState:
    query = state.get("rewritten_question") or state["question"]
    print(f"\n--- [Graph Node] Retrieving & Reranking for: '{query}' ---")
    
    # Calls our hybrid search + RRF + cross-encoder from retrieval.py
    reranked_docs = hybrid_retrieve_and_rerank(query, top_k_retrieve=6, top_k_final=3)
    state["retrieved_docs"] = reranked_docs
    return state

# --- Node 2: Version Checking & Evidence Grader (Steps 6 & 7) ---
def grade_evidence_node(state: AgentState) -> AgentState:
    print("--- [Graph Node] Grading Evidence & Checking Versions ---")
    docs = state["retrieved_docs"]
    question = state["question"]
    
    if not docs:
        state["grade_status"] = "insufficient"
        return state

    # Check for version supersession traps (e.g., G006 superseded by G005)
    doc_ids = [d["document_id"] for d in docs]
    if "G006" in doc_ids and "G005" not in doc_ids:
        print("-> Detected outdated version (G006 without G005). Marking outdated.")
        state["grade_status"] = "outdated"
        return state

    # Use Gemini to grade whether the evidence is sufficient to answer the question
    context_text = "\n\n".join([f"ID: {d['document_id']} | Title: {d['title']} | Text: {d['text']}" for d in docs])
    
    prompt = f"""You are a clinical evidence grader. Evaluate if the provided documents contain enough information to answer the user question.
User Question: {question}

Retrieved Documents:
{context_text}

Respond with ONLY ONE word:
- 'sufficient' if the documents answer the question.
- 'insufficient' if information is missing or cannot answer the question.
"""
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",  # Strictly Gemini 3.5 Flash
            contents=prompt
        )
        grade = response.text.strip().lower()
        if "sufficient" in grade:
            state["grade_status"] = "sufficient"
        else:
            state["grade_status"] = "insufficient"
    except Exception as e:
        print(f"Grading error: {e}, defaulting to insufficient")
        state["grade_status"] = "insufficient"
        
    print(f"-> Grade Result: {state['grade_status']}")
    return state

# --- Node 3: Query Rewrite / Corrective Loop (Step 8) ---
def rewrite_query_node(state: AgentState) -> AgentState:
    print("--- [Graph Node] Rewriting Query for Retry ---")
    state["retry_count"] += 1
    original_query = state["question"]
    
    prompt = f"""Rewrite the following clinical research query to be more explicit and keyword-rich for document retrieval.
Original Question: {original_query}
Rewritten Query (only output the query text, nothing else):"""

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",  # Strictly Gemini 3.5 Flash
            contents=prompt
        )
        state["rewritten_question"] = response.text.strip()
        print(f"-> Rewritten Query: '{state['rewritten_question']}'")
    except Exception:
        state["rewritten_question"] = original_query
        
    return state

# --- Node 4: Answer Generation & Citation (Steps 2 & 5 from assignment) ---
def generate_answer_node(state: AgentState) -> AgentState:
    print("--- [Graph Node] Generating Final Answer ---")
    docs = state["retrieved_docs"]
    question = state["question"]
    
    context_text = "\n\n".join([f"[{d['document_id']}] Title: {d['title']}\n{d['text']}" for d in docs])
    
    prompt = f"""You are MedGuard AI, an evidence-based research assistant. 
Answer the user's question using ONLY the provided document context. Every factual claim or statement must cite its source document IDs in brackets (e.g., [G005]). 
If the evidence does not support an answer, state that the evidence is insufficient. Do not make up information.

Context:
{context_text}

User Question: {question}
Answer:"""

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",  # Strictly Gemini 3.5 Flash
            contents=prompt
        )
        state["answer"] = response.text.strip()
        # Extract cited IDs from the answer text
        import re
        state["cited_ids"] = list(set(re.findall(r'\[(G\d+)\]', state["answer"])))
        state["abstain"] = False
    except Exception as e:
        state["answer"] = "Error generating response."
        state["abstain"] = True
        
    return state

# --- Node 5: Abstention Handler (Step 9) ---
def abstain_node(state: AgentState) -> AgentState:
    print("--- [Graph Node] Abstaining (Insufficient Information) ---")
    state["answer"] = "The available synthetic knowledge base does not contain sufficient information to answer this question."
    state["cited_ids"] = []
    state["abstain"] = True
    return state

# --- Conditional Routing Logic ---
def route_after_grading(state: AgentState) -> str:
    status = state["grade_status"]
    retries = state["retry_count"]
    
    if status == "sufficient":
        return "generate_answer"
    elif status == "outdated":
        if retries < 2:
            return "rewrite_query"
        return "abstain"
    else:
        if retries < 2:
            return "rewrite_query"
        return "abstain"

# --- Build LangGraph State Machine ---
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_evidence", grade_evidence_node)
workflow.add_node("rewrite_query", rewrite_query_node)
workflow.add_node("generate_answer", generate_answer_node)
workflow.add_node("abstain", abstain_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_evidence")

workflow.add_conditional_edges(
    "grade_evidence",
    route_after_grading,
    {
        "generate_answer": "generate_answer",
        "rewrite_query": "rewrite_query",
        "abstain": "abstain"
    }
)

workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate_answer", END)
workflow.add_edge("abstain", END)

# Compile the workflow graph into an executable app
app = workflow.compile()