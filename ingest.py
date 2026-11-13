import json
import chromadb
from google import genai

client = genai.Client()
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection_name = "medguard_docs"
try:
    chroma_client.delete_collection(name=collection_name)
except Exception:
    pass

collection = chroma_client.create_collection(name=collection_name)

# Change this line in ingest.py:
def load_and_populate_chroma(filepath="samples/documents.jsonl"):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    ids, documents, metadatas = [], [], []

    for line in lines:
        if not line.strip():
            continue
        doc = json.loads(line)
        doc_id = doc["document_id"]
        
        ids.append(doc_id)
        documents.append(doc["text"])
        metadatas.append({
            "document_id": doc_id,
            "type": doc.get("type", ""),
            "version": doc.get("version", ""),
            "date": doc.get("date", ""),
            "organization": doc.get("organization", ""),
            "title": doc.get("title", "")
        })

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Successfully ingested {len(ids)} documents into ChromaDB collection '{collection_name}'!")

if __name__ == "__main__":
    load_and_populate_chroma()