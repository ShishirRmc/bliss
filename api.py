"""
FastAPI application serving the recommendation engine and RAG chatbot.

Endpoints:
    GET  /health     — Health check
    POST /recommend  — Service recommendations for a customer
    POST /chatbot    — RAG-powered chatbot with tool-calling
"""

import os
import asyncio
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm_provider import LLMProvider
from recommender import ContentBasedRecommender
from chatbot import ChatbotModel
from rag.ingest import DocumentIngestor
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.retriever import HybridRetriever
from rag.generator import GroundedGenerator


class RecommendRequest(BaseModel):
    customer_id: str = Field(..., description="Customer ID to get recommendations for")
    top_n: int = Field(3, ge=1, le=10, description="Number of recommendations")


class RecommendResponse(BaseModel):
    customer_id: str
    recommendations: list[str]
    confidence_scores: list[float]
    details: list[dict]


class ChatbotRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")
    session_id: str = Field("default", description="Session ID for multi-turn")


class ChatbotResponse(BaseModel):
    response: str
    intent: str
    action_taken: dict | None = None
    sources: list[str] = []


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    provider: str
    vector_store_docs: int


recommender: ContentBasedRecommender | None = None
chatbot: ChatbotModel | None = None
customer_db: set[str] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and initialize RAG pipeline on startup."""
    global recommender, chatbot, customer_db

    print("🚀 Starting Blys AI API...")

    # Load customer data for validation
    data_path = os.path.join(os.path.dirname(__file__), "data", "customer_data.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        customer_db = set(df["Customer_ID"].astype(str).tolist())
        print(f"   ✅ Loaded {len(customer_db)} customers")
    else:
        print(f"   ⚠️  Customer data not found at {data_path}")
        df = None

    # Load or fit recommendation model
    model_path = os.path.join(os.path.dirname(__file__), "models", "recommendation_model.pkl")
    if os.path.exists(model_path):
        recommender = ContentBasedRecommender.load(model_path)
        print("   ✅ Loaded recommendation model from .pkl")
    elif df is not None:
        recommender = ContentBasedRecommender()
        recommender.fit(df)
        print("   ✅ Fitted recommendation model from data")
    else:
        print("   ⚠️  No recommendation model available")

    # Initialize RAG pipeline
    kb_dir = os.path.join(os.path.dirname(__file__), "data", "knowledge_base")
    llm = LLMProvider()
    print(f"   ✅ LLM Provider: {llm.provider}")

    try:
        # Ingest knowledge base (always needed for BM25 index)
        ingestor = DocumentIngestor(chunk_size=512, chunk_overlap=50)
        documents = ingestor.load_documents(kb_dir)
        chunks = ingestor.chunk_documents(documents)
        print(f"   ✅ Ingested {len(documents)} docs → {len(chunks)} chunks")

        # Build embeddings
        embedding_model = EmbeddingModel()

        # Vector store — skip re-embedding if collection already populated
        force_reindex = os.getenv("FORCE_REINDEX", "false").lower() == "true"
        vector_store = VectorStore(collection_name="blys_api", persist_dir="./chroma_db")

        if vector_store.count > 0 and not force_reindex:
            print(f"   ✅ Vector store: reusing {vector_store.count} persisted documents (set FORCE_REINDEX=true to rebuild)")
        else:
            embeddings = embedding_model.embed([c["content"] for c in chunks])
            print(f"   ✅ Generated {len(embeddings)} embeddings ({embedding_model.model_name})")
            vector_store.reset()
            vector_store.add_documents(chunks, embeddings)
            print(f"   ✅ Vector store: {vector_store.count} documents")

        # Retriever
        retriever = HybridRetriever(vector_store, embedding_model, chunks)
        print("   ✅ Hybrid retriever ready (BM25 + Vector + Reranker)")

        # Generator
        generator = GroundedGenerator(llm)

        # Chatbot
        chatbot_path = os.path.join(os.path.dirname(__file__), "models", "chatbot_model.pkl")
        if os.path.exists(chatbot_path):
            chatbot = ChatbotModel.load(
                chatbot_path,
                llm_provider=llm,
                retriever=retriever,
                generator=generator,
            )
            print("Loaded chatbot from .pkl + reconnected RAG")
        else:
            chatbot = ChatbotModel(
                llm_provider=llm,
                retriever=retriever,
                generator=generator,
            )
            print(" Initialized fresh chatbot with RAG pipeline")

    except Exception as e:
        print(f" RAG pipeline error: {e}")
        chatbot = ChatbotModel(llm_provider=llm)
        print("Chatbot initialized without RAG (fallback mode)")

    print("Blys AI API ready!")
    yield
    print("🔴 Shutting down...")


app = FastAPI(
    title="Blys AI API",
    description="AI-powered recommendation engine and RAG chatbot for Blys wellness platform.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    provider = "unknown"
    vector_docs = 0
    try:
        if chatbot and chatbot.llm:
            provider = chatbot.llm.provider
        if chatbot and chatbot.retriever:
            vector_docs = chatbot.retriever.vector_store.count
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        models_loaded=recommender is not None and chatbot is not None,
        provider=provider,
        vector_store_docs=vector_docs,
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """Get service recommendations for a customer."""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommendation model not loaded")

    if customer_db and request.customer_id not in customer_db:
        raise HTTPException(
            status_code=404,
            detail=f"Customer '{request.customer_id}' not found",
        )

    results = recommender.recommend(request.customer_id, top_n=request.top_n)

    return RecommendResponse(
        customer_id=request.customer_id,
        recommendations=[r["service"] for r in results],
        confidence_scores=[r["confidence"] for r in results],
        details=results,
    )


@app.post("/chatbot", response_model=ChatbotResponse)
async def chat(request: ChatbotRequest):
    """Chat with the Blys AI assistant."""
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not loaded")

    # predict() is synchronous — run in a thread to avoid blocking the event loop
    # NOTE: session history is in-process only, not shared across workers
    result = await asyncio.to_thread(
        chatbot.predict,
        user_input=request.message,
        session_id=request.session_id,
    )

    return ChatbotResponse(
        response=result["response"],
        intent=result["intent"],
        action_taken=result.get("action_taken"),
        sources=result.get("sources", []),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
