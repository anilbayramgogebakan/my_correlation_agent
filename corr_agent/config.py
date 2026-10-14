"""Runtime configuration, read from environment variables (and an optional .env file)."""

import os

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

FRED_API_KEY = os.getenv("FRED_API_KEY") or None

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30"))
HTTP_CACHE_TTL = float(os.getenv("HTTP_CACHE_TTL", "3600"))
# Wikimedia asks API clients to send a descriptive User-Agent.
USER_AGENT = os.getenv(
    "HTTP_USER_AGENT",
    "InsDrone-CorrelationAgent/0.1 (proof-of-concept; python-requests)",
)

AGENT_RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "30"))
# Cap on generated tokens per LLM call: a cost/latency guard on free tiers. The final
# explanation is the longest output (~500 tokens).
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1500"))
