# Blys AI Engineer Assessment

## Overview

A production-ready AI backend for the Blys wellness platform, featuring a **content-based recommendation engine**, a **RAG-powered chatbot** with hybrid search and tool-calling, and a **FastAPI serving layer** — all containerised with Docker. Includes a React + Tailwind frontend for interactive exploration.

The chatbot combines a Retrieval-Augmented Generation pipeline (hybrid BM25 + vector search, cross-encoder reranking, grounded generation) for information queries with LLM tool-calling for action intents (reschedule, cancel, book). The recommendation engine uses cosine similarity between customer preference vectors and service feature vectors, with a popularity-based cold-start fallback.

**Backend stack:** Python 3.11, FastAPI, ChromaDB, sentence-transformers, rank-bm25, HuggingFace Transformers, scikit-learn, multi-provider LLM support (OpenAI / Google Gemini / Ollama Cloud).

**Frontend stack:** React 19, Vite, Tailwind CSS v4, GSAP 3.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        FastAPI (api.py)                       │
│  ┌──────────────┐   ┌────────────────────────────────────┐   │
│  │  /recommend  │   │            /chatbot                │   │
│  │              │   │                                    │   │
│  │  Content-    │   │   ┌──────────┐   ┌────────────┐   │   │
│  │  Based       │   │   │  Intent  │──►│  ACTION     │   │   │
│  │  Recommender │   │   │  Router  │   │  Tool-Call  │   │   │
│  │  (cosine     │   │   │  (LLM)   │   │  Handlers   │   │   │
│  │   similarity)│   │   └──────────┘   └────────────┘   │   │
│  │              │   │        │                           │   │
│  │              │   │        ▼ INFO                      │   │
│  │              │   │   ┌──────────────────────────┐     │   │
│  │              │   │   │     RAG Pipeline          │     │   │
│  │              │   │   │  ┌────────┐ ┌──────────┐ │     │   │
│  │              │   │   │  │ BM25   │ │ Vector   │ │     │   │
│  │              │   │   │  │ Search │ │ Search   │ │     │   │
│  │              │   │   │  └───┬────┘ └────┬─────┘ │     │   │
│  │              │   │   │      └──┬─────┬──┘       │     │   │
│  │              │   │   │     RRF Merge            │     │   │
│  │              │   │   │         │                 │     │   │
│  │              │   │   │   Cross-Encoder Rerank   │     │   │
│  │              │   │   │         │                 │     │   │
│  │              │   │   │   Grounded Generation    │     │   │
│  │              │   │   └──────────────────────────┘     │   │
│  └──────────────┘   └────────────────────────────────────┘   │
│                                                              │
│  LLM Provider: OpenAI │ Gemini │ Ollama Cloud (configurable) │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                   React Frontend (frontend/)                  │
│                                                              │
│   Floating pill Navbar  ·  Chat view  ·  Recommend view      │
│   GSAP animations  ·  Midnight Luxe design system            │
└──────────────────────────────────────────────────────────────┘
```

## Setup & Running

### Prerequisites
- Python 3.11+
- Node.js 18+
- An API key for at least one LLM provider

### Backend

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API key and preferred provider

# Generate synthetic data
python data/generate_data.py

# Run the API
python api.py
# → http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

The frontend expects the backend running on `http://localhost:8000`. Make sure the API is up before using the UI.

### Docker (backend only)

```bash
docker build -t blys-assessment .
docker run -p 8000:8000 --env-file .env blys-assessment
```

### Run Evaluation

```bash
python eval/evaluate_rag.py
```

### Run Tests

```bash
pytest tests/ -v
```

## API Documentation

### Health Check

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "models_loaded": true, "provider": "openai", "vector_store_docs": 42}
```

### Recommendations

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "1697", "top_n": 3}'
```

```json
{
  "customer_id": "1697",
  "recommendations": ["Wellness Package", "Couples Massage", "Deep Tissue Massage"],
  "confidence_scores": [0.92, 0.87, 0.81],
  "details": [...]
}
```

### Chatbot

```bash
# Reschedule flow (multi-turn)
curl -X POST http://localhost:8000/chatbot \
  -H "Content-Type: application/json" \
  -d '{"message": "Can I reschedule my booking?", "session_id": "user123"}'

# Information query (RAG-powered)
curl -X POST http://localhost:8000/chatbot \
  -H "Content-Type: application/json" \
  -d '{"message": "How much does a deep tissue massage cost?", "session_id": "user456"}'
```

## Section-by-Section Approach

### Section 1: Customer Behaviour Analysis

Generated 800 synthetic customer records across 5 archetypes (loyalists, occasionals, lapsed high-spenders, new explorers, churned). Ran sentiment analysis using both HuggingFace DistilBERT (local, CPU) and LLM API for comparison. Clustered using K-Means (k=4) with named segments and actionable business recommendations per segment.

**Key decision:** Used a pretrained DistilBERT for sentiment rather than training from scratch — it's a solved problem and a pretrained model outperforms anything trainable on 800 synthetic reviews.

### Section 2: AI-Powered Personalisation Model

Built a content-based filtering engine with a 7-service catalog. Customer preference vectors are weighted averages of their booked service features. Recommendations are ranked by cosine similarity, filtering out already-booked services. Evaluated with Precision@K using leave-last-out split.

**Key decision:** Chose content-based over collaborative filtering because collaborative filtering on synthetic data with 5–7 service types would just memorise the training set. Content-based is more honest and explainable.

### Section 3: NLP Chatbot (RAG + Tool-Calling)

Built a dual-mode chatbot:
- **Information queries** (pricing, policies, FAQs) → RAG pipeline: 4 knowledge base documents → chunked → embedded (sentence-transformers) → ChromaDB → hybrid search (BM25 + vector, RRF merge) → cross-encoder reranking → grounded LLM generation with source citations.
- **Action queries** (reschedule, cancel) → LLM tool-calling with mock action handlers and multi-turn conversation state.

Evaluated against a 25-question golden dataset measuring intent accuracy, retrieval precision, keyword coverage, and LLM-judge faithfulness.

**Key decision:** The brief asks for an "intent classification model saved as .pkl". Rather than training a toy sklearn classifier on 20 examples, I wrapped the LLM-based chatbot in a serialisable class that persists its tool definitions and session state. The LLM IS the classifier — it classifies intent as part of its reasoning with far higher accuracy than any small trained model could achieve on open-ended natural language.

### Section 4: FastAPI API

Three endpoints (`/health`, `/recommend`, `/chatbot`) with Pydantic validation, CORS, proper error handling, and `session_id` support for multi-turn conversations. The API initialises the full RAG pipeline on startup.

**Key decision:** Added `session_id` to the chatbot endpoint (not in the original brief) because without it you can't maintain conversation history across HTTP requests, which breaks the multi-turn reschedule flow they specifically asked for.

### Section 5: React Frontend

A dark-themed single-page app built with React 19, Vite, Tailwind CSS v4, and GSAP. Features a floating pill navbar, a centered chat widget with RAG source citations, and a recommendation panel with animated confidence bars. Design system based on the "Midnight Luxe" preset (Obsidian/Champagne palette, Inter + Playfair Display + JetBrains Mono typography).

## Design Decisions

1. **Multi-provider LLM abstraction** — Supports OpenAI, Gemini, and Ollama Cloud via a single `LLMProvider` class selected by environment variable. This makes the project portable and avoids vendor lock-in.

2. **Hybrid search with reranking** — BM25 catches keyword matches (e.g., exact pricing figures), vector search catches semantic similarity (e.g., "change my appointment" → rescheduling). RRF merges both, and a cross-encoder reranker ensures the final ranking is precise.

3. **Dual sentiment approach** — HuggingFace for demonstrating traditional NLP skills (runs on CPU, no API cost), LLM API as a comparison baseline. Both are shown in the notebook.

4. **Section-aware chunking** — Documents are split on markdown headers first, preserving semantic coherence within chunks, rather than blindly splitting on character count.

## Known Limitations & What I'd Do With More Time

**Recommendation model:**
- On synthetic data, the model has artificially good metrics because the data has no real signal. In production, I'd replace this with a two-tower neural model trained on implicit feedback signals (browse time, scroll depth, rebooking patterns) and augment with location and time-of-day features.
- Cold-start problem: currently handled with a popularity fallback. In production, I'd add a brief onboarding preference survey for new users.

**Chatbot:**
- Session history grows unbounded in memory. In production, I'd use Redis-backed session storage with TTL expiry.
- The RAG pipeline re-indexes on every API startup. In production, I'd persist the ChromaDB and only re-index when knowledge base documents change.
- No conversation memory summarisation — long conversations will hit the context window limit. I'd implement a sliding window with summarisation.

**API:**
- No authentication or rate limiting. In production: JWT auth, per-user rate limits, structured logging with correlation IDs for distributed tracing.
- No async handling for LLM calls — would add background tasks for long-running operations.

**Evaluation:**
- The golden dataset is small (25 questions). In production, I'd build a continuous evaluation loop: log production queries, subsample 1% for LLM-judge scoring, and monitor faithfulness/relevancy metrics over time.


## Overview

A production-ready AI backend for the Blys wellness platform, featuring a **content-based recommendation engine**, a **RAG-powered chatbot** with hybrid search and tool-calling, and a **FastAPI serving layer** — all containerised with Docker.

The chatbot combines a Retrieval-Augmented Generation pipeline (hybrid BM25 + vector search, cross-encoder reranking, grounded generation) for information queries with LLM tool-calling for action intents (reschedule, cancel, book). The recommendation engine uses cosine similarity between customer preference vectors and service feature vectors, with a popularity-based cold-start fallback.

**Tech stack:** Python 3.11, FastAPI, ChromaDB, sentence-transformers, rank-bm25, HuggingFace Transformers, scikit-learn, multi-provider LLM support (OpenAI / Google Gemini / Ollama Cloud).

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        FastAPI (api.py)                       │
│  ┌──────────────┐   ┌────────────────────────────────────┐   │
│  │  /recommend  │   │            /chatbot                │   │
│  │              │   │                                    │   │
│  │  Content-    │   │   ┌──────────┐   ┌────────────┐   │   │
│  │  Based       │   │   │  Intent  │──►│  ACTION     │   │   │
│  │  Recommender │   │   │  Router  │   │  Tool-Call  │   │   │
│  │  (cosine     │   │   │  (LLM)   │   │  Handlers   │   │   │
│  │   similarity)│   │   └──────────┘   └────────────┘   │   │
│  │              │   │        │                           │   │
│  │              │   │        ▼ INFO                      │   │
│  │              │   │   ┌──────────────────────────┐     │   │
│  │              │   │   │     RAG Pipeline          │     │   │
│  │              │   │   │  ┌────────┐ ┌──────────┐ │     │   │
│  │              │   │   │  │ BM25   │ │ Vector   │ │     │   │
│  │              │   │   │  │ Search │ │ Search   │ │     │   │
│  │              │   │   │  └───┬────┘ └────┬─────┘ │     │   │
│  │              │   │   │      └──┬─────┬──┘       │     │   │
│  │              │   │   │     RRF Merge            │     │   │
│  │              │   │   │         │                 │     │   │
│  │              │   │   │   Cross-Encoder Rerank   │     │   │
│  │              │   │   │         │                 │     │   │
│  │              │   │   │   Grounded Generation    │     │   │
│  │              │   │   └──────────────────────────┘     │   │
│  └──────────────┘   └────────────────────────────────────┘   │
│                                                              │
│  LLM Provider: OpenAI │ Gemini │ Ollama Cloud (configurable) │
└──────────────────────────────────────────────────────────────┘
```

## Setup & Running

### Prerequisites
- Python 3.11+
- An API key for at least one LLM provider

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd bliss

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API key and preferred provider
```

### Generate Data

```bash
python data/generate_data.py
```

### Run the API

```bash
python api.py
# or
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker build -t blys-assessment .
docker run -p 8000:8000 --env-file .env blys-assessment
```

### Run Evaluation

```bash
python eval/evaluate_rag.py
```

## API Documentation

### Health Check

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "models_loaded": true, "provider": "openai", "vector_store_docs": 42}
```

### Recommendations

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "1001", "top_n": 3}'
```

```json
{
  "customer_id": "1001",
  "recommendations": ["Wellness Package", "Couples Massage", "Deep Tissue Massage"],
  "confidence_scores": [0.92, 0.87, 0.81],
  "details": [...]
}
```

### Chatbot

```bash
# Reschedule flow (multi-turn)
curl -X POST http://localhost:8000/chatbot \
  -H "Content-Type: application/json" \
  -d '{"message": "Can I reschedule my booking?", "session_id": "user123"}'

# Follow-up
curl -X POST http://localhost:8000/chatbot \
  -H "Content-Type: application/json" \
  -d '{"message": "Yes", "session_id": "user123"}'

# Provide date
curl -X POST http://localhost:8000/chatbot \
  -H "Content-Type: application/json" \
  -d '{"message": "30 Mar 2025 10 am", "session_id": "user123"}'

# Information query (RAG-powered)
curl -X POST http://localhost:8000/chatbot \
  -H "Content-Type: application/json" \
  -d '{"message": "How much does a deep tissue massage cost?", "session_id": "user456"}'
```

## Section-by-Section Approach

### Section 1: Customer Behaviour Analysis

Generated 800 synthetic customer records across 5 archetypes (loyalists, occasionals, lapsed high-spenders, new explorers, churned). Ran sentiment analysis using both HuggingFace DistilBERT (local, CPU) and LLM API for comparison. Clustered using K-Means (k=4) with named segments and actionable business recommendations per segment.

**Key decision:** Used a pretrained DistilBERT for sentiment rather than training from scratch — it's a solved problem and a pretrained model outperforms anything trainable on 800 synthetic reviews.

### Section 2: AI-Powered Personalisation Model

Built a content-based filtering engine with a 7-service catalog. Customer preference vectors are weighted averages of their booked service features. Recommendations are ranked by cosine similarity, filtering out already-booked services. Evaluated with Precision@K using leave-last-out split.

**Key decision:** Chose content-based over collaborative filtering because collaborative filtering on synthetic data with 5–7 service types would just memorise the training set. Content-based is more honest and explainable.

### Section 3: NLP Chatbot (RAG + Tool-Calling)

Built a dual-mode chatbot:
- **Information queries** (pricing, policies, FAQs) → RAG pipeline: 4 knowledge base documents → chunked → embedded (sentence-transformers) → ChromaDB → hybrid search (BM25 + vector, RRF merge) → cross-encoder reranking → grounded LLM generation with source citations.
- **Action queries** (reschedule, cancel) → LLM tool-calling with mock action handlers and multi-turn conversation state.

Evaluated against a 25-question golden dataset measuring intent accuracy, retrieval precision, keyword coverage, and LLM-judge faithfulness.

**Key decision:** The brief asks for an "intent classification model saved as .pkl". Rather than training a toy sklearn classifier on 20 examples, I wrapped the LLM-based chatbot in a serialisable class that persists its tool definitions and session state. The LLM IS the classifier — it classifies intent as part of its reasoning with far higher accuracy than any small trained model could achieve on open-ended natural language.

### Section 4: FastAPI API

Three endpoints (`/health`, `/recommend`, `/chatbot`) with Pydantic validation, CORS, proper error handling, and `session_id` support for multi-turn conversations. The API initialises the full RAG pipeline on startup.

**Key decision:** Added `session_id` to the chatbot endpoint (not in the original brief) because without it you can't maintain conversation history across HTTP requests, which breaks the multi-turn reschedule flow they specifically asked for.

## Design Decisions

1. **Multi-provider LLM abstraction** — Supports OpenAI, Gemini, and Ollama Cloud via a single `LLMProvider` class selected by environment variable. This makes the project portable and avoids vendor lock-in.

2. **Hybrid search with reranking** — BM25 catches keyword matches (e.g., exact pricing figures), vector search catches semantic similarity (e.g., "change my appointment" → rescheduling). RRF merges both, and a cross-encoder reranker ensures the final ranking is precise.

3. **Dual sentiment approach** — HuggingFace for demonstrating traditional NLP skills (runs on CPU, no API cost), LLM API as a comparison baseline. Both are shown in the notebook.

4. **Section-aware chunking** — Documents are split on markdown headers first, preserving semantic coherence within chunks, rather than blindly splitting on character count.

## Known Limitations & What I'd Do With More Time

**Recommendation model:**
- On synthetic data, the model has artificially good metrics because the data has no real signal. In production, I'd replace this with a two-tower neural model trained on implicit feedback signals (browse time, scroll depth, rebooking patterns) and augment with location and time-of-day features.
- Cold-start problem: currently handled with a popularity fallback. In production, I'd add a brief onboarding preference survey for new users.

**Chatbot:**
- Session history grows unbounded in memory. In production, I'd use Redis-backed session storage with TTL expiry (I've implemented this pattern at BoomConsole for concurrent multi-agent sessions).
- The RAG pipeline re-indexes on every API startup. In production, I'd persist the ChromaDB and only re-index when knowledge base documents change.
- No conversation memory summarisation — long conversations will hit the context window limit. I'd implement a sliding window with summarisation.

**API:**
- No authentication or rate limiting. In production: JWT auth, per-user rate limits, structured logging with correlation IDs for distributed tracing.
- No async handling for LLM calls — would add background tasks for long-running operations.

**Evaluation:**
- The golden dataset is small (25 questions). In production, I'd build a continuous evaluation loop: log production queries, subsample 1% for LLM-judge scoring, and monitor faithfulness/relevancy metrics over time (I built this pattern at Fatafat Sewa for monitoring 2,000+ daily user queries).
"# bliss" 
