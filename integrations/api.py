from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging

# Import the compiled LangGraph app from main.py
from core.main import app as langgraph_app

app = FastAPI(
    title="MedGuard AI API",
    description="Evidence-based clinical research assistant powered by LangGraph and Gemini 3.5 Flash.",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    cited_ids: List[str]
    abstain: bool

@app.get("/")
def read_root():
    """Welcome route so visiting http://127.0.0.1:8000 shows a clear message instead of 404."""
    return {
        "message": "Welcome to MedGuard AI API!",
        "documentation": "Go to /docs for the interactive Swagger UI to test the /ask endpoint."
    }

@app.post("/ask", response_model=QueryResponse)
def ask_question(payload: QueryRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    initial_state = {
        "question": payload.question,
        "rewritten_question": "",
        "retrieved_docs": [],
        "grade_status": "",
        "retry_count": 0,
        "answer": "",
        "cited_ids": [],
        "abstain": False
    }
    
    try:
        # Invoke your LangGraph state machine workflow
        result = langgraph_app.invoke(initial_state)
        return {
            "question": result["question"],
            "answer": result["answer"],
            "cited_ids": result["cited_ids"],
            "abstain": result["abstain"]
        }
    except Exception as e:
        logging.error(f"Error during graph execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)