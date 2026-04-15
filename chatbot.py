"""
RAG-powered chatbot with intent routing and tool-calling.

Combines a RAG pipeline (hybrid search, reranking, grounded generation)
for information queries with tool-calling for action intents.
"""

import json
import pickle
import re

from llm_provider import LLMProvider
from rag.retriever import HybridRetriever
from rag.generator import GroundedGenerator


# ── Mock action handlers ─────────────────────────────────────────────────────


def reschedule_booking(booking_id: str, new_datetime: str) -> dict:
    """Mock: Send reschedule request to therapist."""
    return {
        "status": "pending_confirmation",
        "message": (
            f"Reschedule request sent to your therapist for {new_datetime}. "
            f"You'll be notified once it's confirmed."
        ),
        "booking_id": booking_id,
    }


def cancel_booking(booking_id: str) -> dict:
    """Mock: Cancel a booking."""
    return {
        "status": "cancelled",
        "message": (
            f"Your booking {booking_id} has been cancelled. "
            f"A refund will be processed within 5-10 business days "
            f"(subject to our cancellation policy)."
        ),
        "booking_id": booking_id,
    }


def get_pricing(service_type: str) -> dict:
    """Retained for backwards compatibility — pricing is handled via RAG."""
    return {
        "status": "info_request",
        "message": f"Looking up pricing information for {service_type}...",
        "service_type": service_type,
    }


TOOL_FUNCTIONS = {
    "reschedule_booking": reschedule_booking,
    "cancel_booking": cancel_booking,
}

TOOL_DEFINITIONS = [
    {
        "name": "reschedule_booking",
        "description": (
            "Reschedule an existing booking to a new date and time. "
            "Use this when the customer wants to change the date/time of their booking."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "booking_id": {
                    "type": "string",
                    "description": "The booking ID to reschedule. If not provided by the customer, use 'LATEST' as default.",
                },
                "new_datetime": {
                    "type": "string",
                    "description": "The new date and time for the booking, e.g. '30 Mar 2025 10:00 AM'.",
                },
            },
            "required": ["booking_id", "new_datetime"],
        },
    },
    {
        "name": "cancel_booking",
        "description": (
            "Cancel an existing booking. Always confirm with the customer before calling this."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "booking_id": {
                    "type": "string",
                    "description": "The booking ID to cancel. If not provided, use 'LATEST'.",
                },
            },
            "required": ["booking_id"],
        },
    },
]

CHATBOT_SYSTEM_PROMPT = """\
You are a friendly and professional AI assistant for Blys, an on-demand \
wellness platform. You help customers with booking, rescheduling, \
cancellations, pricing inquiries, and general questions about Blys services.

BEHAVIOUR RULES:
1. For ACTION requests (reschedule, cancel, new booking):
   - Ask clarifying questions if you don't have all the information needed.
   - For rescheduling: you MUST get the new date and time before calling the \
reschedule tool. Ask for it if not provided. Always use the word "reschedule" \
in your response so the customer knows what action is being taken.
   - For cancellation: ALWAYS confirm with the customer before proceeding.
   - Use the provided tools to execute actions.

2. For INFORMATION requests (pricing, policies, how-to questions):
   - You will be provided with context from the Blys knowledge base.
   - Answer based on that context. Do not make up information.

3. For NEW BOOKINGS:
   - Guide the customer to book through the Blys app.
   - You cannot create bookings directly.

4. TONE: Warm, professional, concise. Use the customer's name if known.

5. If you cannot help with something, offer to connect them with the support team.
"""



class ChatbotModel:
    """RAG-powered chatbot with intent routing and tool-calling."""

    INTENTS = [
        "reschedule_booking",
        "cancel_booking",
        "booking_new",
        "pricing_inquiry",
        "general_inquiry",
    ]

    ACTION_INTENTS = {"reschedule_booking", "cancel_booking", "booking_new"}
    INFO_INTENTS = {"pricing_inquiry", "general_inquiry"}

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        retriever: HybridRetriever | None = None,
        generator: GroundedGenerator | None = None,
    ):
        self.llm = llm_provider or LLMProvider()
        self.retriever = retriever
        self.generator = generator
        self.sessions: dict[str, list[dict]] = {}  # session_id → history
        self.tools = TOOL_DEFINITIONS

    def predict(self, user_input: str, session_id: str = "default") -> dict:
        """
        Process a user message and return a response.

        Returns:
            {
                "response": str,
                "intent": str,
                "action_taken": dict | None,
                "sources": list[str],
                "retrieval_scores": list[float]
            }
        """
        # Get or create session state
        if session_id not in self.sessions:
            self.sessions[session_id] = {"history": [], "active_intent": None}
        session = self.sessions[session_id]
        history = session["history"]

        # Add user message
        history.append({"role": "user", "content": user_input})

        intent = self._classify_intent(user_input, history, session)

        action_taken = None
        sources = []
        retrieval_scores = []

        if intent in self.ACTION_INTENTS:
            # Track active intent so multi-turn continuity doesn't rely on keyword matching
            session["active_intent"] = intent
            # Action flow: use LLM with tool-calling
            response, action_taken = self._handle_action(history)
            # Clear active intent once a tool call completes the action
            if action_taken is not None:
                session["active_intent"] = None
        elif intent in self.INFO_INTENTS and self.retriever and self.generator:
            # Information flow: RAG pipeline
            response, sources, retrieval_scores = self._handle_info(
                user_input, history
            )
        else:
            # Fallback: direct LLM response (no RAG available)
            response = self._handle_fallback(history)

        # Add assistant response to history
        history.append({"role": "assistant", "content": response})

        # Trim history to avoid context overflow (keep last 20 turns)
        if len(history) > 20:
            session["history"] = history[-20:]

        return {
            "response": response,
            "intent": intent,
            "action_taken": action_taken,
            "sources": sources,
            "retrieval_scores": retrieval_scores,
        }

    def _classify_intent(self, user_input: str, history: list[dict], session: dict) -> str:
        """
        Classify intent using the LLM.
        Active action intents persist across multi-turn conversations without
        re-classifying, so the LLM is only called on fresh turns.
        """
        # Continue active action intent across multi-turn conversations
        active = session.get("active_intent")
        if active and active in self.ACTION_INTENTS:
            return active

        # LLM-based classification
        classify_prompt = (
            f"Classify this customer message into exactly one of these intents:\n"
            f"reschedule_booking, cancel_booking, booking_new, pricing_inquiry, general_inquiry\n\n"
            f"Rules:\n"
            f"- reschedule_booking: customer wants to PERFORM a reschedule action on their booking\n"
            f"- cancel_booking: customer wants to PERFORM a cancellation action on their booking\n"
            f"- booking_new: customer wants to create a new booking\n"
            f"- pricing_inquiry: asks about prices, costs, or service types\n"
            f"- general_inquiry: asks about policies, FAQs, or how things work (including questions ABOUT cancellation/rescheduling policies)\n\n"
            f"Message: \"{user_input}\"\n\n"
            f"Reply with ONLY the intent name."
        )

        result = self.llm.complete(classify_prompt)
        intent = result.strip().lower().replace('"', "").replace("'", "")

        if intent not in self.INTENTS:
            for valid_intent in self.INTENTS:
                if valid_intent in intent:
                    return valid_intent
            return "general_inquiry"

        return intent

    def _handle_action(self, history: list[dict]) -> tuple[str, dict | None]:
        """Handle action intents via LLM tool-calling."""
        messages = [{"role": "system", "content": CHATBOT_SYSTEM_PROMPT}]
        messages.extend(history)

        # Call LLM with tools
        result = self.llm.chat(messages, tools=self.tools)

        if result["tool_calls"]:
            # Execute the tool call
            tool_call = result["tool_calls"][0]
            func_name = tool_call["function"]["name"]
            func_args = tool_call["function"]["arguments"]

            if func_name in TOOL_FUNCTIONS:
                action_result = TOOL_FUNCTIONS[func_name](**func_args)

                # If it's a pricing request, route through RAG instead
                if action_result.get("status") == "info_request":
                    if self.retriever and self.generator:
                        query = history[-1]["content"]
                        response, _, _ = self._handle_info(query, history)
                        return response, None

                # Generate a natural response incorporating the action result
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"[Tool called: {func_name}({func_args})]",
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "name": func_name,
                        "content": json.dumps(action_result),
                    }
                )

                followup = self.llm.chat(messages)
                return followup["content"], {
                    "tool": func_name,
                    "args": func_args,
                    "result": action_result,
                }

        # No tool call — LLM is asking a clarifying question
        return result["content"], None

    def _handle_info(
        self, query: str, history: list[dict]
    ) -> tuple[str, list[str], list[float]]:
        """Handle information intents via RAG retrieval + grounded generation."""
        # Retrieve relevant chunks
        chunks = self.retriever.retrieve(query, top_k=5)

        # Extract scores
        retrieval_scores = [
            c.get("rerank_score", c.get("score", 0)) for c in chunks
        ]

        # Generate grounded response
        result = self.generator.generate(
            query=query,
            context_chunks=chunks,
            conversation_history=history[:-1],  # exclude current query
        )

        return result["response"], result["sources"], retrieval_scores

    def _handle_fallback(self, history: list[dict]) -> str:
        """Direct LLM response without RAG (fallback)."""
        messages = [{"role": "system", "content": CHATBOT_SYSTEM_PROMPT}]
        messages.extend(history)
        result = self.llm.chat(messages)
        return result["content"]

    def save(self, filepath: str = "models/chatbot_model.pkl"):
        """
        Save the chatbot configuration (not the LLM client).
        On load, the LLMProvider is re-initialized from .env.
        """
        state = {
            "tools": self.tools,
            "sessions": self.sessions,  # dict[session_id, {"history": [...], "active_intent": ...}]
            "class": "ChatbotModel",
            "version": "1.1",
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(
        cls,
        filepath: str = "models/chatbot_model.pkl",
        llm_provider: LLMProvider | None = None,
        retriever: HybridRetriever | None = None,
        generator: GroundedGenerator | None = None,
    ) -> "ChatbotModel":
        """Load a saved chatbot and reconnect to LLM + RAG."""
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        bot = cls(
            llm_provider=llm_provider,
            retriever=retriever,
            generator=generator,
        )
        bot.tools = state["tools"]
        # Migrate old format (list) to new format (dict with history + active_intent)
        raw_sessions = state.get("sessions", {})
        for sid, val in raw_sessions.items():
            if isinstance(val, list):
                bot.sessions[sid] = {"history": val, "active_intent": None}
            else:
                bot.sessions[sid] = val
        return bot
