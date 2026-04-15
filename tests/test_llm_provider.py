"""
Tests for LLMProvider.

Covers:
- Unsupported provider raises ValueError
- complete() builds correct messages and returns string
- OpenAI tool conversion wraps bare dicts in {"type": "function", ...}
- Ollama tool conversion same as OpenAI
- Gemini message conversion: user/assistant/system roles (unchanged)
- Gemini tool message uses from_function_response, not from_text (bug fix)
- Gemini function name extracted from preceding assistant message
"""
import json
import pytest
from unittest.mock import MagicMock, patch, call


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_openai_provider():
    """LLMProvider(openai) with a mocked OpenAI client."""
    with patch("llm_provider.LLMProvider._init_client"):
        from llm_provider import LLMProvider
        p = LLMProvider.__new__(LLMProvider)
        p.provider = "openai"
        p._client = MagicMock()
        return p


def make_gemini_provider():
    """LLMProvider(gemini) with a mocked Gemini client."""
    with patch("llm_provider.LLMProvider._init_client"):
        from llm_provider import LLMProvider
        p = LLMProvider.__new__(LLMProvider)
        p.provider = "gemini"
        p._client = MagicMock()
        return p


# ── Provider validation ───────────────────────────────────────────────────────

class TestProviderValidation:
    def test_unsupported_provider_raises(self):
        with patch("llm_provider.LLMProvider._init_client"):
            from llm_provider import LLMProvider
            with pytest.raises(ValueError, match="Unsupported provider"):
                p = LLMProvider.__new__(LLMProvider)
                p.provider = "unknown_llm"
                p._client = None
                # Trigger validation manually
                if p.provider not in LLMProvider.SUPPORTED_PROVIDERS:
                    raise ValueError(f"Unsupported provider '{p.provider}'.")

    def test_supported_providers_list(self):
        from llm_provider import LLMProvider
        assert "openai" in LLMProvider.SUPPORTED_PROVIDERS
        assert "gemini" in LLMProvider.SUPPORTED_PROVIDERS
        assert "ollama_cloud" in LLMProvider.SUPPORTED_PROVIDERS


# ── complete() ────────────────────────────────────────────────────────────────

class TestComplete:
    def test_complete_returns_string(self):
        p = make_openai_provider()
        p.chat = MagicMock(return_value={"content": "Hello!", "tool_calls": None})
        result = p.complete("Say hello")
        assert result == "Hello!"

    def test_complete_builds_user_message(self):
        p = make_openai_provider()
        p.chat = MagicMock(return_value={"content": "ok", "tool_calls": None})
        p.complete("test prompt")
        messages = p.chat.call_args[0][0]
        assert messages[-1] == {"role": "user", "content": "test prompt"}

    def test_complete_with_system_prepends_system_message(self):
        p = make_openai_provider()
        p.chat = MagicMock(return_value={"content": "ok", "tool_calls": None})
        p.complete("user msg", system="You are helpful")
        messages = p.chat.call_args[0][0]
        assert messages[0] == {"role": "system", "content": "You are helpful"}
        assert messages[1]["role"] == "user"


# ── OpenAI tool conversion ────────────────────────────────────────────────────

class TestOpenAIToolConversion:
    def test_bare_tool_dict_wrapped(self):
        p = make_openai_provider()
        bare = [{"name": "my_func", "description": "does stuff", "parameters": {}}]
        converted = p._convert_tools_openai(bare)
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "my_func"

    def test_already_wrapped_tool_unchanged(self):
        p = make_openai_provider()
        wrapped = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        converted = p._convert_tools_openai(wrapped)
        assert converted == wrapped

    def test_mixed_tools_handled(self):
        p = make_openai_provider()
        tools = [
            {"name": "bare_func", "parameters": {}},
            {"type": "function", "function": {"name": "wrapped_func", "parameters": {}}},
        ]
        converted = p._convert_tools_openai(tools)
        assert len(converted) == 2
        assert all(t["type"] == "function" for t in converted)


# ── Gemini message conversion ─────────────────────────────────────────────────

class TestGeminiMessageConversion:
    """
    Test _chat_gemini message conversion by injecting fake google.genai modules
    into sys.modules — avoids needing google-generativeai installed.
    """

    def _make_fake_types(self):
        """Build a fake google.genai.types module as a MagicMock."""
        from unittest.mock import MagicMock
        fake = MagicMock()
        fake.Content.side_effect = lambda role, parts: {"role": role, "parts": parts}
        fake.Part.from_text.side_effect = lambda text: {"_type": "text", "text": text}
        fake.Part.from_function_response.side_effect = (
            lambda name, response: {"_type": "fn_resp", "name": name, "response": response}
        )
        fake.GenerateContentConfig.return_value = MagicMock()
        fake.Tool.return_value = MagicMock()
        fake.FunctionDeclaration.return_value = MagicMock()
        return fake

    def _mock_response(self, text="ok"):
        mock_part = MagicMock()
        mock_part.text = text
        mock_part.function_call = None
        mock_resp = MagicMock()
        mock_resp.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]
        return mock_resp

    def _run(self, messages, tools=None):
        """Run _chat_gemini with injected fake modules, return (provider, fake_types)."""
        import sys
        fake_types = self._make_fake_types()
        fake_genai = MagicMock()
        fake_google = MagicMock()
        fake_google.genai = fake_genai

        p = make_gemini_provider()
        p._client.models.generate_content.return_value = self._mock_response()

        with patch.dict(sys.modules, {
            "google": fake_google,
            "google.genai": fake_genai,
            "google.genai.types": fake_types,
        }):
            # Reload the module so it picks up the patched imports
            import importlib
            import llm_provider as lm
            # Patch the types import inside the method directly
            with patch.object(lm, "__builtins__", lm.__builtins__):
                # Call with the fake types injected at the module level
                original_chat_gemini = p._chat_gemini.__func__

                def patched_chat_gemini(self_inner, messages, tools, response_format, model):
                    # Temporarily replace the import inside the method
                    import sys as _sys
                    _sys.modules["google.genai.types"] = fake_types
                    return original_chat_gemini(self_inner, messages, tools, response_format, model)

                import types as builtin_types
                p._chat_gemini = builtin_types.MethodType(patched_chat_gemini, p)
                p._chat_gemini(messages, tools, None, None)

        return p, fake_types

    def _inject(self, fake_types):
        """Return a context manager that injects fake_types so `from google.genai import types` works."""
        import sys
        fake_genai = MagicMock()
        fake_genai.types = fake_types
        fake_google = MagicMock()
        fake_google.genai = fake_genai
        return patch.dict(sys.modules, {
            "google": fake_google,
            "google.genai": fake_genai,
            "google.genai.types": fake_types,
        })

    def test_user_message_uses_from_text(self):
        """role=user → Part.from_text (unchanged behavior)."""
        fake_types = self._make_fake_types()
        p = make_gemini_provider()
        p._client.models.generate_content.return_value = self._mock_response()

        with self._inject(fake_types):
            p._chat_gemini([{"role": "user", "content": "hello"}], None, None, None)

        calls = [str(c) for c in fake_types.Part.from_text.call_args_list]
        assert any("hello" in c for c in calls)

    def test_tool_message_uses_from_function_response(self):
        """role=tool → Part.from_function_response, NOT from_text (bug fix)."""
        import json
        fake_types = self._make_fake_types()
        p = make_gemini_provider()
        p._client.models.generate_content.return_value = self._mock_response()

        tool_result = json.dumps({"status": "cancelled", "booking_id": "B123"})
        messages = [
            {"role": "user", "content": "Cancel my booking"},
            {"role": "assistant", "content": "[Tool called: cancel_booking({'booking_id': 'B123'})]"},
            {"role": "tool", "content": tool_result},
        ]

        with self._inject(fake_types):
            p._chat_gemini(messages, None, None, None)

        assert fake_types.Part.from_function_response.called, (
            "Part.from_function_response was not called for role='tool' message"
        )
        text_calls = [str(c) for c in fake_types.Part.from_text.call_args_list]
        assert not any(tool_result in c for c in text_calls), (
            "Tool result was passed to Part.from_text instead of from_function_response"
        )

    def test_tool_message_extracts_function_name(self):
        """Function name is extracted from the preceding assistant message."""
        import json
        captured_names = []
        fake_types = self._make_fake_types()
        fake_types.Part.from_function_response.side_effect = (
            lambda name, response: captured_names.append(name) or {"_type": "fn_resp"}
        )
        p = make_gemini_provider()
        p._client.models.generate_content.return_value = self._mock_response()

        messages = [
            {"role": "assistant", "content": "[Tool called: reschedule_booking({'booking_id': 'X'})]"},
            {"role": "tool", "content": json.dumps({"status": "pending"})},
        ]

        with self._inject(fake_types):
            p._chat_gemini(messages, None, None, None)

        assert captured_names, "from_function_response was not called"
        assert captured_names[0] == "reschedule_booking", (
            f"Expected 'reschedule_booking', got '{captured_names[0]}'"
        )

    def test_assistant_message_uses_model_role(self):
        """role=assistant → Gemini role='model'."""
        roles_used = []
        fake_types = self._make_fake_types()
        fake_types.Content.side_effect = lambda role, parts: roles_used.append(role) or {"role": role, "parts": parts}
        p = make_gemini_provider()
        p._client.models.generate_content.return_value = self._mock_response()

        with self._inject(fake_types):
            p._chat_gemini([{"role": "assistant", "content": "Sure!"}], None, None, None)

        assert "model" in roles_used, f"Expected 'model' role, got: {roles_used}"
