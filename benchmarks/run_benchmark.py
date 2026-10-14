"""Run the same test questions on several LLMs and record behaviour, tokens and latency.

    python -m benchmarks.run_benchmark                       # all cloud models, all cases
    python -m benchmarks.run_benchmark --models ollama:qwen2.5:7b
    python -m benchmarks.run_benchmark --cases A B

Each run appends one JSON line to benchmarks/results.jsonl; summarize with benchmarks.report.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from corr_agent.agent import run_agent
from corr_agent.data_store import DatasetStore
from corr_agent.llm import get_chat_model

RESULTS = Path(__file__).with_name("results.jsonl")

CASES = {
    "A": {
        "name": "Weather × Wikipedia + forecast",
        "question": "Do hot days in New York drive interest in ice cream? Forecast next week.",
    },
    "B": {
        "name": "FRED × FRED, mixed frequency + lag",
        "question": "Do oil prices drive US gasoline prices? Is there a lag?",
    },
    "C": {
        "name": "Unavailable data (proxy trap)",
        "question": "Do high pollen days in Berlin increase sales of allergy medication?",
    },
}

CLOUD_MODELS = [
    "gemini:gemini-3.5-flash-lite",
    "gemini:gemini-3.5-flash",
    "groq:openai/gpt-oss-20b",
    "groq:openai/gpt-oss-120b",
    "groq:qwen/qwen3.8-27b",
]


def run_case(spec: str, case_id: str) -> dict:
    provider, model = spec.split(":", 1)
    store = DatasetStore()
    calls, errors, answer, usage = [], [], "", {}
    for ev in run_agent(CASES[case_id]["question"], get_chat_model(provider, model), store):
        if ev.kind == "decision":
            calls.append({"tool": ev.data["tool"], "args": ev.data["args"]})
        elif ev.kind == "error":
            errors.append(ev.message)
        elif ev.kind == "final":
            answer, usage = ev.message, ev.data
    analysis = store.analyses[-1].to_llm_dict() if store.analyses else None
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "case": case_id, "provider": provider, "model": model,
        "usage": usage, "tool_calls": calls, "errors": errors,
        "datasets": [d.label for d in store.list()],
        "analysis": analysis, "answer": answer,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=CLOUD_MODELS, help="provider:model ...")
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--pause", type=float, default=5.0, help="seconds between runs (free-tier rate limits)")
    args = parser.parse_args()

    for spec in args.models:
        for case_id in args.cases:
            print(f"▶ {spec} · case {case_id}", flush=True)
            try:
                record = run_case(spec, case_id)
            except Exception as e:  # e.g. model not available on this key
                record = {"timestamp": datetime.now().isoformat(timespec="seconds"), "case": case_id,
                          "provider": spec.split(":", 1)[0], "model": spec.split(":", 1)[1],
                          "usage": {}, "tool_calls": [], "errors": [f"{type(e).__name__}: {e}"],
                          "datasets": [], "analysis": None, "answer": ""}
            with RESULTS.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            u = record["usage"]
            print(f"  {u.get('seconds')} s · {u.get('llm_calls')} LLM calls · {u.get('input_tokens')} in / "
                  f"{u.get('output_tokens')} out · tools {len(record['tool_calls'])} · errors {len(record['errors'])}",
                  flush=True)
            if args.pause and spec.split(":", 1)[0] != "ollama":
                time.sleep(args.pause)


if __name__ == "__main__":
    main()
