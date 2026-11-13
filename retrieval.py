import os
# Prevent Windows threading/multiprocessing deadlocks with NumPy/Torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
from rank_bm25 import BM25Okapi
import chromadb
from sentence_transformers import CrossEncoder

# Initialize Persistent Chroma client and collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="medguard_docs")

# Load raw documents for BM25 and metadata lookup
def load_raw_docs(filepath="documents.jsonl"):
    docs = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                docs[doc["document_id"]] = doc
    return docs

raw_docs_dict = load_raw_docs()
raw_docs_list = list(raw_docs_dict.values())
tokenized_corpus = [doc["text"].lower().split() for doc in raw_docs_list]
bm25 = BM25Okapi(tokenized_corpus)
doc_ids_list = [doc["document_id"] for doc in raw_docs_list]

# Initialize Cross-Encoder model for Reranking
print("Loading Cross-Encoder model for reranking...")
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def keyword_search(query: str, top_k=5):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [doc_ids_list[i] for i in top_indices]

def semantic_search(query: str, top_k=5):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results["ids"][0] if results["ids"] else []

def reciprocal_rank_fusion(semantic_ids, keyword_ids, k=60):
    fused_scores = {}
    for rank, doc_id in enumerate(semantic_ids):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
    for rank, doc_id in enumerate(keyword_ids):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
    sorted_docs = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    return sorted_docs

def hybrid_retrieve_and_rerank(query: str, top_k_retrieve=6, top_k_final=3):
    sem_ids = semantic_search(query, top_k=top_k_retrieve)
    kw_ids = keyword_search(query, top_k=top_k_retrieve)
    fused_ids = reciprocal_rank_fusion(sem_ids, kw_ids)[:top_k_retrieve]
    
    candidates = [raw_docs_dict[doc_id] for doc_id in fused_ids if doc_id in raw_docs_dict]
    if not candidates:
        return []
        
    pairs = [[query, doc["text"]] for doc in candidates]
    scores = reranker_model.predict(pairs)
    
    for i, doc in enumerate(candidates):
        doc["relevance_score"] = float(scores[i])
        
    sorted_candidates = sorted(candidates, key=lambda x: x["relevance_score"], reverse=True)
    return sorted_candidates[:top_k_final]