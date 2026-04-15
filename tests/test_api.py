"""
Tests for the FastAPI endpoints.

Uses TestClient with mocked global state — no real LLM, DB, or vector store needed.

Covers:
- GET /health returns correct shape
- POST /recommend happy path, 404 for unknown customer, 503 when model not loaded
- POST /chatbot happy path, 503 when chatbot not loaded
- POST /chatbot uses asyncio.to_thread (bug fix — non-blocking)
- Response models match Pydantic schemas
"""
import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient


# ── App fixture ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_recommender():
    rec = MagicMock()
    rec.recommend.return_value = [
        {"service": "Facial", "service_id": "SVC002", "confidence": 0.85,
         "category": "face", "avg_price": 95.0},
        {"service": "Wellness Package", "service_id": "SVC003", "confidence": 0.72,
         "category": "combo", "avg_price": 179.0},
        {"service": "Manicure", "service_id": "SVC004", "confidence": 0.60,
         "category": "nails", "avg_price": 55.0},
    ]
    return rec


@pytest.fixture
def mock_chatbot():
    bot = MagicMock()
    bot.llm = MagicMock()
    bot.llm.provider = "openai"
    bot.retriever = MagicMock()
    bot.retriever.vector_store.count = 42
    bot.predict.return_value = {
        "response": "I can help with that!",
        "intent": "general_inquiry",
        "action_taken": None,
        "sources": ["pricing.md"],
        "retrieval_scores": [0.9],
    }
    return bot


@pytest.fixture
def client(mock_recommender, mock_chatbot):
    """TestClient with mocked global state, bypassing lifespan startup."""
    import api as api_module

    api_module.recommender = mock_recommender
    api_module.chatbot = mock_chatbot
    api_module.customer_db = {"C001", "C002", "C003"}

    # Instantiate without context manager — starlette 0.36 + httpx 0.28
    # broke the `app=` kwarg when used as a context manager.
    c = TestClient(api_module.app, raise_server_exceptions=True)
    yield c

    # Cleanup
    api_module.recommender = None
    api_module.chatbot = None
    api_module.customer_db = set()


# ── GET /health ───────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_response_shape(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "models_loaded" in data
        assert "provider" in data
        assert "vector_store_docs" in data

    def test_models_loaded_true(self, client):
        data = client.get("/health").json()
        assert data["models_loaded"] is True

    def test_provider_reported(self, client):
        data = client.get("/health").json()
        assert data["provider"] == "openai"

    def test_vector_store_docs_reported(self, client):
        data = client.get("/health").json()
        assert data["vector_store_docs"] == 42


# ── POST /recommend ───────────────────────────────────────────────────────────

class TestRecommend:
    def test_happy_path(self, client):
        resp = client.post("/recommend", json={"customer_id": "C001", "top_n": 3})
        assert resp.status_code == 200

    def test_response_shape(self, client):
        data = client.post("/recommend", json={"customer_id": "C001", "top_n": 3}).json()
        assert "customer_id" in data
        assert "recommendations" in data
        assert "confidence_scores" in data
        assert "details" in data

    def test_customer_id_echoed(self, client):
        data = client.post("/recommend", json={"customer_id": "C001", "top_n": 3}).json()
        assert data["customer_id"] == "C001"

    def test_recommendations_count(self, client):
        data = client.post("/recommend", json={"customer_id": "C001", "top_n": 3}).json()
        assert len(data["recommendations"]) == 3

    def test_confidence_scores_count_matches(self, client):
        data = client.post("/recommend", json={"customer_id": "C001", "top_n": 3}).json()
        assert len(data["confidence_scores"]) == len(data["recommendations"])

    def test_unknown_customer_returns_404(self, client):
        resp = client.post("/recommend", json={"customer_id": "UNKNOWN_999", "top_n": 3})
        assert resp.status_code == 404

    def test_top_n_validation_min(self, client):
        resp = client.post("/recommend", json={"customer_id": "C001", "top_n": 0})
        assert resp.status_code == 422  # Pydantic validation

    def test_top_n_validation_max(self, client):
        resp = client.post("/recommend", json={"customer_id": "C001", "top_n": 11})
        assert resp.status_code == 422

    def test_503_when_recommender_not_loaded(self, mock_chatbot):
        import api as api_module
        api_module.recommender = None
        api_module.chatbot = mock_chatbot
        api_module.customer_db = set()
        c = TestClient(api_module.app)
        resp = c.post("/recommend", json={"customer_id": "C001", "top_n": 3})
        assert resp.status_code == 503
        api_module.recommender = None


# ── POST /chatbot ─────────────────────────────────────────────────────────────

class TestChatbot:
    def test_happy_path(self, client):
        resp = client.post("/chatbot", json={"message": "Hello", "session_id": "s1"})
        assert resp.status_code == 200

    def test_response_shape(self, client):
        data = client.post("/chatbot", json={"message": "Hello", "session_id": "s1"}).json()
        assert "response" in data
        assert "intent" in data
        assert "action_taken" in data
        assert "sources" in data

    def test_response_content(self, client):
        data = client.post("/chatbot", json={"message": "Hello", "session_id": "s1"}).json()
        assert data["response"] == "I can help with that!"
        assert data["intent"] == "general_inquiry"

    def test_sources_list(self, client):
        data = client.post("/chatbot", json={"message": "Hello", "session_id": "s1"}).json()
        assert isinstance(data["sources"], list)

    def test_empty_message_rejected(self, client):
        resp = client.post("/chatbot", json={"message": "", "session_id": "s1"})
        assert resp.status_code == 422

    def test_503_when_chatbot_not_loaded(self, mock_recommender):
        import api as api_module
        api_module.recommender = mock_recommender
        api_module.chatbot = None
        api_module.customer_db = set()
        c = TestClient(api_module.app)
        resp = c.post("/chatbot", json={"message": "Hi", "session_id": "s1"})
        assert resp.status_code == 503
        api_module.chatbot = None

    def test_predict_called_with_correct_args(self, client, mock_chatbot):
        client.post("/chatbot", json={"message": "Test message", "session_id": "my_session"})
        mock_chatbot.predict.assert_called_once_with(
            user_input="Test message",
            session_id="my_session",
        )


# ── asyncio.to_thread usage (Bug 6 fix) ──────────────────────────────────────

class TestAsyncEndpoint:
    def test_chat_endpoint_uses_asyncio_to_thread(self):
        """The /chatbot endpoint must use asyncio.to_thread, not a direct call."""
        import inspect
        import api as api_module
        source = inspect.getsource(api_module.chat)
        assert "asyncio.to_thread" in source, (
            "chat() endpoint must use asyncio.to_thread to avoid blocking the event loop"
        )

    def test_asyncio_imported_in_api(self):
        import api as api_module
        assert hasattr(api_module, "asyncio") or "asyncio" in dir(api_module), (
            "asyncio must be imported in api.py"
        )
