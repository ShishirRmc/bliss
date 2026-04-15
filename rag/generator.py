"""
Grounded response generator for the RAG pipeline.

Takes retrieved context chunks and generates a response that is
strictly grounded in the provided information, with source citations.
"""

from llm_provider import LLMProvider


GENERATION_SYSTEM_PROMPT = """\
You are a helpful customer support assistant for Blys, an on-demand wellness \
platform that connects customers with therapists for in-home massage, facial, \
and wellness services.

INSTRUCTIONS:
- Answer the customer's question using ONLY the provided context.
- If the context does not contain enough information to answer, say: \
"I don't have enough information to answer that. Let me connect you with our \
support team for further assistance."
- Be warm, professional, and concise.
- When citing specific details (prices, policies, timeframes), mention the \
source naturally, e.g. "According to our pricing guide..." or "Per our \
cancellation policy...".
- Do NOT make up information that is not in the context.
- Format prices in dollars ($).
"""


class GroundedGenerator:
    """Generate responses grounded in retrieved context."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    def generate(
        self,
        query: str,
        context_chunks: list[dict],
        conversation_history: list[dict] | None = None,
    ) -> dict:
        """
        Generate a grounded response from retrieved context.

        Args:
            query: The user's question.
            context_chunks: Retrieved chunks with content and source.
            conversation_history: Previous conversation messages.

        Returns:
            {"response": str, "sources": list[str]}
        """
        # Format context
        context_text = self._format_context(context_chunks)

        # Build messages
        messages = [{"role": "system", "content": GENERATION_SYSTEM_PROMPT}]

        # Add conversation history (if any)
        if conversation_history:
            for msg in conversation_history[-6:]:  # last 6 turns to avoid overflow
                messages.append(msg)

        # Add context + query
        user_message = (
            f"CONTEXT (retrieved from Blys knowledge base):\n"
            f"---\n{context_text}\n---\n\n"
            f"CUSTOMER QUESTION: {query}"
        )
        messages.append({"role": "user", "content": user_message})

        # Generate
        result = self.llm.chat(messages)
        response = result["content"]

        # Collect unique sources
        sources = list(set(c["source"] for c in context_chunks))

        return {
            "response": response,
            "sources": sources,
        }

    def _format_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a context string."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "unknown")
            score = chunk.get("rerank_score", chunk.get("score", 0))
            parts.append(
                f"[Source: {source} | Relevance: {score:.2f}]\n{chunk['content']}"
            )
        return "\n\n".join(parts)
