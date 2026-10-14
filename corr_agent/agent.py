"""The orchestrator: a LangChain/LangGraph tool-calling agent whose steps are streamed as events.

Usage from the command line:
    python -m corr_agent.agent "Do hot days in New York drive interest in ice cream?" --provider ollama
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from corr_agent import config
from corr_agent.data_store import DatasetStore
from corr_agent.prompts import build_system_prompt
from corr_agent.tools import build_tools


@dataclass
class AgentEvent:
    kind: str  # "start" | "decision" | "step" | "tool_result" | "error" | "usage" | "final"
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: datetime = field(default_factory=datetime.now)


def build_agent(model: BaseChatModel, store: DatasetStore):
    tools = build_tools(store)
    return create_agent(model, tools, system_prompt=build_system_prompt(fred_enabled=bool(config.FRED_API_KEY)))


def _format_call(name: str, args: dict[str, Any]) -> str:
    shown = ", ".join(f"{k}={v!r}" for k, v in args.items() if v is not None)
    return f"{name}({shown})"


def _summarize_tool_result(msg: ToolMessage) -> tuple[str, str]:
    try:
        payload = json.loads(msg.content) if isinstance(msg.content, str) else {}
    except json.JSONDecodeError:
        payload = {}
    if "error" in payload:
        return "error", f"{msg.name} returned an error: {payload['error']}"
    if "dataset_id" in payload:
        return "tool_result", (f"{msg.name} → {payload['dataset_id']} · {payload.get('n_points')} points "
                               f"({payload.get('start')} → {payload.get('end')})")
    if "results" in payload:
        found = [r.get("title") or r.get("series_id") for r in payload["results"]][:5]
        return "tool_result", f"{msg.name} → {found}"
    if "pearson_r" in payload:
        return "tool_result", (f"{msg.name} → r = {payload['pearson_r']}, ρ = {payload['spearman_rho']}, "
                               f"p = {payload['pearson_p_value']}, n = {payload['n_points']}")
    return "tool_result", f"{msg.name} finished"


def _add_usage(usage: dict[str, Any], msg: AIMessage) -> None:
    meta = msg.usage_metadata or {}
    usage["llm_calls"] += 1
    usage["input_tokens"] += meta.get("input_tokens", 0)
    usage["output_tokens"] += meta.get("output_tokens", 0)
    usage["reasoning_tokens"] += (meta.get("output_token_details") or {}).get("reasoning", 0)


def run_agent(question: str, model: BaseChatModel, store: DatasetStore,
              history: list[BaseMessage] | None = None) -> Iterator[AgentEvent]:
    """Run the agent on one question, yielding events as they happen. The last event is "final"."""
    agent = build_agent(model, store)
    messages = [*(history or []), HumanMessage(question)]
    yield AgentEvent("start", f"Agent started with {getattr(model, 'model', type(model).__name__)}")

    final_text = ""
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
             "tool_calls": 0, "tool_errors": 0}
    started = time.perf_counter()
    stream = agent.stream({"messages": messages}, stream_mode=["updates", "custom"],
                          config={"recursion_limit": config.AGENT_RECURSION_LIMIT})
    try:
        for mode, chunk in stream:
            if mode == "custom":
                kind = "error" if chunk.get("level") == "error" else "step"
                yield AgentEvent(kind, chunk.get("message", ""), chunk)
                continue
            for update in chunk.values():
                for msg in (update or {}).get("messages", []):
                    if isinstance(msg, AIMessage):
                        _add_usage(usage, msg)
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for call in msg.tool_calls:
                            yield AgentEvent("decision", "LLM decided to call " + _format_call(call["name"],
                                                                                                call["args"]),
                                             {"tool": call["name"], "args": call["args"]})
                    elif isinstance(msg, AIMessage):
                        final_text = msg.text or final_text
                    elif isinstance(msg, ToolMessage):
                        kind, text = _summarize_tool_result(msg)
                        usage["tool_calls"] += 1
                        usage["tool_errors"] += kind == "error"
                        yield AgentEvent(kind, text, {"tool": msg.name})
    except Exception as e:  # LLM/provider failures (bad key, model missing, recursion limit, ...)
        yield AgentEvent("error", f"Agent run failed: {type(e).__name__}: {e}")
        final_text = final_text or f"Sorry, the agent failed: {e}"

    usage["seconds"] = round(time.perf_counter() - started, 1)
    yield AgentEvent("usage", (f"{usage['llm_calls']} LLM calls · {usage['input_tokens']:,} input / "
                               f"{usage['output_tokens']:,} output tokens · {usage['tool_calls']} tool calls "
                               f"({usage['tool_errors']} failed) · {usage['seconds']} s"), usage)
    yield AgentEvent("final", final_text or "The agent finished without a textual answer.", usage)


def main() -> None:
    from corr_agent.llm import get_chat_model

    parser = argparse.ArgumentParser(description="Ask the correlation agent a question.")
    parser.add_argument("question")
    parser.add_argument("--provider", default=config.LLM_PROVIDER, choices=["gemini", "groq", "ollama"])
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    store = DatasetStore()
    for event in run_agent(args.question, get_chat_model(args.provider, args.model), store):
        if event.kind == "final":
            print("\n" + "=" * 80 + "\n" + event.message)
        else:
            print(f"[{event.ts:%H:%M:%S}] {event.kind:<11} {event.message}")


if __name__ == "__main__":
    main()
