import os

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from google import genai

# -----------------------------
# App + config
# -----------------------------

app = FastAPI()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash") if GOOGLE_API_KEY else None
client = MongoClient(MONGO_URI) if MONGO_URI else None
collection = client["edulearn"]["curriculum"] if client else None
embedding_model = None

# -----------------------------
# Startup
# -----------------------------

@app.on_event("startup")
async def startup_event():
    global embedding_model

    if not MONGO_URI:
        print("MONGO_URI not set, skipping DB connection")

    try:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedding model loaded")
    except Exception as exc:
        embedding_model = None
        print(f"Embedding model unavailable: {exc}")

# -----------------------------
# Request schema
# -----------------------------

class QuestionRequest(BaseModel):
    question: str

# -----------------------------
# Core logic
# -----------------------------

def retrieve_context(question: str) -> str:
    if embedding_model is None:
        raise RuntimeError("Embedding model not available")
    if collection is None:
        raise RuntimeError("Mongo collection not available")

    question_embedding = embedding_model.encode([question])

    documents = list(collection.find())
    texts = [doc.get("content", "") for doc in documents if doc.get("content")]

    if not texts:
        return ""

    embeddings = embedding_model.encode(texts)

    n_neighbors = min(3, len(texts))
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(embeddings)

    _, indices = nn.kneighbors(question_embedding)

    context = ""
    for i in indices[0]:
        context += texts[i] + "\n"

    return context

def generate_answer(question: str) -> str:
    if model is None:
        raise RuntimeError("GOOGLE_API_KEY not configured")

    context = retrieve_context(question)

    prompt = f"""
Answer the question using the curriculum context.

Context:
{context}

Question:
{question}
"""

    response = model.generate_content(prompt)
    return response.text

# -----------------------------
# API endpoints
# -----------------------------

@app.get("/")
def health():
    return {
        "status": "running",
        "model_ready": embedding_model is not None,
        "db_ready": collection is not None,
    }

@app.post("/ask")
def ask_bot(request: QuestionRequest):
    try:
        answer = generate_answer(request.question)
        return {"response": answer}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to generate answer") from exc

# -----------------------------
# Run server
# -----------------------------

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("🚀 EduLearn AI Chatbot API")
    print("=" * 60)
    print(f"📝 Documentation: http://localhost:8000/docs")
    print(f"❤️  Health Check: http://localhost:8000/health")
    print(f"💬 Ask Endpoint: POST http://localhost:8000/ask")
    print("=" * 60)
uvicorn.run(app, host="0.0.0.0", port=PORT)
