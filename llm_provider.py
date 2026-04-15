"""
Multi-provider LLM abstraction layer.

Supports OpenAI, Google Gemini, and Ollama Cloud behind a single interface.
Provider is selected via LLM_PROVIDER in .env.
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()


class LLMProvider:
    """Unified interface for OpenAI, Gemini, and Ollama Cloud."""

    SUPPORTED_PROVIDERS = ("openai", "gemini", "ollama_cloud")

    def __init__(self, provider: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
        if self.provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider '{self.provider}'. "
                f"Choose from: {self.SUPPORTED_PROVIDERS}"
            )
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.provider == "openai":
            from openai import OpenAI

            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        elif self.provider == "gemini":
            from google import genai

            self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

        elif self.provider == "ollama_cloud":
            from ollama import Client

            self._client = Client(
                host="https://ollama.com",
                headers={
                    "Authorization": "Bearer " + (os.getenv("OLLAMA_API_KEY") or "")
                },
            )

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        model: str | None = None,
    ) -> dict:
        """
        Send a chat request through the active provider.

        Args:
            messages: List of {"role": str, "content": str} dicts.
            tools: Optional tool/function definitions for tool-calling.
            response_format: Optional structured output schema.
            model: Override the default model for this provider.

        Returns:
            {"content": str, "tool_calls": list[dict] | None}
        """
        if self.provider == "openai":
            return self._chat_openai(messages, tools, response_format, model)
        elif self.provider == "gemini":
            return self._chat_gemini(messages, tools, response_format, model)
        elif self.provider == "ollama_cloud":
            return self._chat_ollama(messages, tools, response_format, model)

    def _chat_openai(self, messages, tools, response_format, model) -> dict:
        model = model or "gpt-4o-mini"
        kwargs = {"model": model, "messages": messages}

        if tools:
            kwargs["tools"] = self._convert_tools_openai(tools)
            kwargs["tool_choice"] = "auto"
        if response_format:
            kwargs["response_format"] = response_format

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]

        return {"content": msg.content or "", "tool_calls": tool_calls}

    def _convert_tools_openai(self, tools: list[dict]) -> list[dict]:
        """Ensure tools are in OpenAI function-calling format."""
        openai_tools = []
        for tool in tools:
            if "type" in tool and tool["type"] == "function":
                openai_tools.append(tool)
            else:
                openai_tools.append(
                    {
                        "type": "function",
                        "function": tool,
                    }
                )
        return openai_tools

    def _chat_gemini(self, messages, tools, response_format, model) -> dict:
        from google.genai import types

        model = model or "gemini-2.0-flash"

        # Convert messages to Gemini format
        gemini_contents = []
        system_instruction = None
        last_function_name: str | None = None  # track last called function for tool responses

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_instruction = content
            elif role == "user":
                gemini_contents.append(
                    types.Content(
                        role="user", parts=[types.Part.from_text(text=content)]
                    )
                )
            elif role == "assistant":
                # Track function name from tool-call annotations so we can use it
                # when the subsequent tool result message arrives
                if "[Tool called:" in content:
                    try:
                        # Format: "[Tool called: func_name({...})]"
                        fname = content.split("[Tool called:")[1].split("(")[0].strip()
                        last_function_name = fname
                    except Exception:
                        pass
                gemini_contents.append(
                    types.Content(
                        role="model", parts=[types.Part.from_text(text=content)]
                    )
                )
            elif role == "tool":
                # Tool result — must use function_response part so Gemini can
                # correlate the result with the preceding function call.
                try:
                    result_dict = json.loads(content) if isinstance(content, str) else content
                except (json.JSONDecodeError, TypeError):
                    result_dict = {"result": content}
                func_name = msg.get("name") or last_function_name or "unknown_function"
                gemini_contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=func_name,
                                response=result_dict,
                            )
                        ],
                    )
                )

        # Build config
        config_kwargs = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        # Convert tools to Gemini format
        gemini_tools = None
        if tools:
            function_declarations = []
            for tool in tools:
                func = tool if "name" in tool else tool.get("function", tool)
                decl = types.FunctionDeclaration(
                    name=func["name"],
                    description=func.get("description", ""),
                    parameters=func.get("parameters", {}),
                )
                function_declarations.append(decl)
            gemini_tools = [types.Tool(function_declarations=function_declarations)]
            config_kwargs["tools"] = gemini_tools

        if response_format:
            config_kwargs["response_mime_type"] = "application/json"

        config = types.GenerateContentConfig(**config_kwargs)

        response = self._client.models.generate_content(
            model=model, contents=gemini_contents, config=config
        )

        # Parse response
        candidate = response.candidates[0]
        content_text = ""
        tool_calls = None

        for part in candidate.content.parts:
            if part.text:
                content_text += part.text
            elif part.function_call:
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    {
                        "id": f"call_{part.function_call.name}",
                        "function": {
                            "name": part.function_call.name,
                            "arguments": dict(part.function_call.args)
                            if part.function_call.args
                            else {},
                        },
                    }
                )

        return {"content": content_text, "tool_calls": tool_calls}

    def _chat_ollama(self, messages, tools, response_format, model) -> dict:
        model = model or "gpt-oss:120b"
        kwargs = {"model": model, "messages": messages}

        if tools:
            kwargs["tools"] = self._convert_tools_ollama(tools)
        if response_format:
            kwargs["format"] = "json"

        response = self._client.chat(**kwargs)
        msg = response["message"]

        tool_calls = None
        if msg.get("tool_calls"):
            tool_calls = [
                {
                    "id": f"call_{tc['function']['name']}",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"]
                        if isinstance(tc["function"]["arguments"], dict)
                        else json.loads(tc["function"]["arguments"]),
                    },
                }
                for tc in msg["tool_calls"]
            ]

        return {"content": msg.get("content", ""), "tool_calls": tool_calls}

    def _convert_tools_ollama(self, tools: list[dict]) -> list[dict]:
        """Convert tools to Ollama format (OpenAI-compatible)."""
        ollama_tools = []
        for tool in tools:
            if "type" in tool and tool["type"] == "function":
                ollama_tools.append(tool)
            else:
                ollama_tools.append({"type": "function", "function": tool})
        return ollama_tools

    def complete(self, prompt: str, system: str | None = None, **kwargs) -> str:
        """Simple text completion — returns just the content string."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = self.chat(messages, **kwargs)
        return result["content"]

    def __repr__(self) -> str:
        return f"LLMProvider(provider='{self.provider}')"
