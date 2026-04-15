"""
Tests for ChatbotModel.

Covers:
- Session state initialisation
- Intent classification (fresh session uses LLM)
- Active intent persists across turns without keyword matching (bug fix)
- Active intent cleared after tool call completes
- New intent in fresh turn overrides nothing
- predict() response shape
- save/load round-trip with session migration
- Fallback path when no RAG components
"""
import json
import pickle
import tempfile
import os
import pytest
from unittest.mock import MagicMock, patch


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_mock_llm(intent_response="general_inquiry", chat_response="Hello!", tool_calls=None):
    """Return a mock LLMProvider."""
    llm = MagicMock()
    llm.provider = "openai"
    llm.complete.return_value = intent_response
    llm.chat.return_value = {"content": chat_response, "tool_calls": tool_calls}
    return llm


@pytest.fixture
def bot():
    """ChatbotModel with a mock LLM and no RAG."""
    from chatbot import ChatbotModel
    llm = make_mock_llm()
    return ChatbotModel(llm_provider=llm)


# ── Session initialisation ────────────────────────────────────────────────────

class TestSessionInit:
    def test_new_session_created_on_first_predict(self, bot):
        bot.predict("Hi", session_id="s1")
        assert "s1" in bot.sessions
        assert "history" in bot.sessions["s1"]
        assert "active_intent" in bot.sessions["s1"]

    def test_history_grows_with_turns(self, bot):
        bot.predict("Hello", session_id="s1")
        bot.predict("How are you?", session_id="s1")
        history = bot.sessions["s1"]["history"]
        # 2 user + 2 assistant messages
        assert len(history) == 4

    def test_separate_sessions_are_independent(self, bot):
        bot.predict("Hi", session_id="A")
        bot.predict("Hey", session_id="B")
        assert bot.sessions["A"]["history"] != bot.sessions["B"]["history"]


# ── Intent classification ─────────────────────────────────────────────────────

class TestIntentClassification:
    def test_fresh_session_uses_llm(self, bot):
        bot.llm.complete.return_value = "pricing_inquiry"
        result = bot.predict("How much does a massage cost?", session_id="fresh")
        assert result["intent"] == "pricing_inquiry"
        bot.llm.complete.assert_called_once()

    def test_invalid_llm_response_falls_back_to_general(self, bot):
        bot.llm.complete.return_value = "nonsense_intent_xyz"
        result = bot.predict("Something weird", session_id="s1")
        assert result["intent"] == "general_inquiry"

    def test_fuzzy_match_on_partial_intent(self, bot):
        bot.llm.complete.return_value = "this is a reschedule_booking request"
        result = bot.predict("I want to reschedule", session_id="s1")
        assert result["intent"] == "reschedule_booking"

    @pytest.mark.parametrize("intent", [
        "reschedule_booking", "cancel_booking", "booking_new",
        "pricing_inquiry", "general_inquiry",
    ])
    def test_all_valid_intents_accepted(self, bot, intent):
        bot.llm.complete.return_value = intent
        result = bot.predict("test", session_id=f"s_{intent}")
        assert result["intent"] == intent


# ── Multi-turn intent continuity (Bug 3 fix) ──────────────────────────────────

class TestMultiTurnIntentContinuity:
    def test_active_intent_persists_without_keywords(self):
        """
        After classifying reschedule_booking, subsequent turns must return
        reschedule_booking even if the LLM would classify them differently.
        This validates the bug fix: no keyword matching on assistant text.
        """
        from chatbot import ChatbotModel
        llm = make_mock_llm()
        bot = ChatbotModel(llm_provider=llm)

        # Turn 1: user asks to reschedule → LLM classifies as reschedule_booking
        llm.complete.return_value = "reschedule_booking"
        # No tool call yet — LLM asks a clarifying question
        llm.chat.return_value = {"content": "Which day works for you?", "tool_calls": None}
        r1 = bot.predict("I want to reschedule my booking", session_id="s1")
        assert r1["intent"] == "reschedule_booking"
        assert bot.sessions["s1"]["active_intent"] == "reschedule_booking"

        # Turn 2: user provides date — LLM would say general_inquiry but active_intent overrides
        llm.complete.return_value = "general_inquiry"
        llm.chat.return_value = {"content": "Done!", "tool_calls": None}
        r2 = bot.predict("Next Tuesday at 10am", session_id="s1")
        assert r2["intent"] == "reschedule_booking", (
            "Active intent should persist regardless of LLM classification"
        )
        # LLM.complete should NOT have been called for intent classification this turn
        # (active_intent short-circuits it)
        assert llm.complete.call_count == 1  # only called on turn 1

    def test_active_intent_cleared_after_tool_call(self):
        """Once a tool call completes, active_intent is cleared."""
        from chatbot import ChatbotModel
        llm = make_mock_llm()
        bot = ChatbotModel(llm_provider=llm)

        # Turn 1: reschedule intent set
        llm.complete.return_value = "reschedule_booking"
        llm.chat.return_value = {
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "function": {
                    "name": "reschedule_booking",
                    "arguments": {"booking_id": "LATEST", "new_datetime": "Tuesday 10am"},
                },
            }],
        }
        # Second chat call (follow-up after tool) returns a plain response
        llm.chat.side_effect = [
            {"content": "", "tool_calls": [{"id": "c1", "function": {"name": "reschedule_booking", "arguments": {"booking_id": "LATEST", "new_datetime": "Tue 10am"}}}]},
            {"content": "Your booking has been rescheduled!", "tool_calls": None},
        ]
        r1 = bot.predict("Reschedule to Tuesday 10am", session_id="s1")
        assert r1["action_taken"] is not None
        assert bot.sessions["s1"]["active_intent"] is None

    def test_cancel_intent_persists(self):
        """cancel_booking active intent also persists across turns."""
        from chatbot import ChatbotModel
        llm = make_mock_llm()
        bot = ChatbotModel(llm_provider=llm)

        llm.complete.return_value = "cancel_booking"
        llm.chat.return_value = {"content": "Are you sure?", "tool_calls": None}
        bot.predict("Cancel my booking", session_id="s1")
        assert bot.sessions["s1"]["active_intent"] == "cancel_booking"

        # Next turn: LLM would say general_inquiry, but active_intent overrides
        llm.complete.return_value = "general_inquiry"
        llm.chat.return_value = {"content": "Cancelled.", "tool_calls": None}
        r2 = bot.predict("Yes, go ahead", session_id="s1")
        assert r2["intent"] == "cancel_booking"

    def test_fresh_session_no_active_intent(self, bot):
        assert bot.sessions == {}
        bot.predict("Hello", session_id="new")
        # active_intent should be None for a non-action intent
        assert bot.sessions["new"]["active_intent"] is None


# ── predict() response shape ──────────────────────────────────────────────────

class TestPredictResponseShape:
    def test_response_keys_present(self, bot):
        result = bot.predict("Hello", session_id="s1")
        assert "response" in result
        assert "intent" in result
        assert "action_taken" in result
        assert "sources" in result
        assert "retrieval_scores" in result

    def test_response_is_string(self, bot):
        result = bot.predict("Hello", session_id="s1")
        assert isinstance(result["response"], str)

    def test_sources_is_list(self, bot):
        result = bot.predict("Hello", session_id="s1")
        assert isinstance(result["sources"], list)

    def test_action_taken_none_when_no_tool_call(self, bot):
        bot.llm.complete.return_value = "general_inquiry"
        result = bot.predict("What services do you offer?", session_id="s1")
        assert result["action_taken"] is None

    def test_history_trimmed_at_20(self, bot):
        """History should not grow beyond 20 messages."""
        for i in range(15):
            bot.predict(f"Message {i}", session_id="trim_test")
        history = bot.sessions["trim_test"]["history"]
        assert len(history) <= 20


# ── save / load ───────────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_round_trip_preserves_sessions(self):
        from chatbot import ChatbotModel
        llm = make_mock_llm()
        bot = ChatbotModel(llm_provider=llm)
        bot.predict("Hello", session_id="persist_test")

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            bot.save(path)
            loaded = ChatbotModel.load(path, llm_provider=llm)
            assert "persist_test" in loaded.sessions
        finally:
            os.unlink(path)

    def test_old_format_migrated(self):
        """Loading a pkl with old list-based sessions migrates to new dict format."""
        from chatbot import ChatbotModel
        llm = make_mock_llm()

        # Simulate old-format pickle
        old_state = {
            "tools": [],
            "sessions": {"old_session": [{"role": "user", "content": "hi"}]},
            "class": "ChatbotModel",
            "version": "1.0",
        }
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(old_state, f)
            path = f.name
        try:
            loaded = ChatbotModel.load(path, llm_provider=llm)
            session = loaded.sessions["old_session"]
            assert isinstance(session, dict)
            assert "history" in session
            assert "active_intent" in session
            assert session["active_intent"] is None
        finally:
            os.unlink(path)
