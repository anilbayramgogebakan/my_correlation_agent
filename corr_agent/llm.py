"""Chat model factory: one place that knows about providers. Everything else sees a BaseChatModel."""

from __future__ import annotations

import requests
from langchain_core.language_models import BaseChatModel

from corr_agent import config

PROVIDERS = ("gemini", "groq", "ollama")


def get_chat_model(provider: str | None = None, model: str | None = None,
                   temperature: float = 0.0) -> BaseChatModel:
    provider = (provider or config.LLM_PROVIDER).lower()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey")
        # Gemini 3 models use fixed sampling defaults and warn if temperature is set, so it is not passed.
        return ChatGoogleGenerativeAI(model=model or config.GEMINI_MODEL, google_api_key=config.GEMINI_API_KEY,
                                      max_output_tokens=config.LLM_MAX_TOKENS)
    if provider == "groq":
        from langchain_groq import ChatGroq

        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys")
        model = model or config.GROQ_MODEL
        # Reasoning models (gpt-oss, qwen3) spend output tokens on thinking first; keep that short so the
        # answer fits in max_tokens and the free-tier rate limit lasts.
        reasoning = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
        return ChatGroq(model=model, api_key=config.GROQ_API_KEY, temperature=temperature,
                        max_tokens=config.LLM_MAX_TOKENS, **reasoning)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # A larger context window than Ollama's default keeps the tool schemas and history in view.
        return ChatOllama(model=model or config.OLLAMA_MODEL, base_url=config.OLLAMA_HOST,
                          temperature=temperature, num_ctx=16384, num_predict=config.LLM_MAX_TOKENS)
    raise ValueError(f"Unknown provider {provider!r}; choose one of {PROVIDERS}")


def list_ollama_models() -> list[str]:
    """Models installed in the local Ollama server; empty if it is not reachable."""
    try:
        resp = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=2)
        resp.raise_for_status()
    except requests.RequestException:
        return []
    return sorted(m["name"] for m in resp.json().get("models", []))


def list_groq_models() -> list[str]:
    """Chat models in Groq's (frequently changing) catalogue; empty if no key or unreachable."""
    if not config.GROQ_API_KEY:
        return []
    try:
        resp = requests.get("https://api.groq.com/openai/v1/models", timeout=5,
                            headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"})
        resp.raise_for_status()
    except requests.RequestException:
        return []
    skip = ("whisper", "tts", "guard", "embed", "orpheus")
    return sorted(m["id"] for m in resp.json().get("data", []) if not any(k in m["id"] for k in skip))


def list_gemini_models() -> list[str]:
    """Gemini models that support generateContent; empty if no key or the API is unreachable."""
    if not config.GEMINI_API_KEY:
        return []
    try:
        from google import genai

        client = genai.Client(api_key=config.GEMINI_API_KEY)
        names = [m.name.removeprefix("models/") for m in client.models.list()
                 if "generateContent" in (m.supported_actions or [])]
    except Exception:
        return []
    return sorted(n for n in names if n.startswith("gemini") and "embedding" not in n)
