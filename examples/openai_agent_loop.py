"""Minimal agent loop with InstructionGuard + OpenAI-compatible API.

Works with any OpenAI-compatible provider (OpenAI, Ollama, LiteLLM, vLLM, etc.).
No framework dependencies — just urllib and InstructionGuard.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/openai_agent_loop.py
"""
from __future__ import annotations

import json
import os
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

from instructionguard import InstructionGuard


def chat(messages: list[dict], model: str = "gpt-4o-mini") -> str:
    """Minimal OpenAI-compatible chat call using only stdlib."""
    api_key = os.environ["OPENAI_API_KEY"]
    endpoint = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") + "/chat/completions"
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.7}).encode()
    req = Request(endpoint, data=body, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def main() -> None:
    with TemporaryDirectory(prefix="instructionguard-openai-") as tmpdir:
        guard = InstructionGuard(agent_id="openai-agent", storage_path=tmpdir)

        # 1. Protect critical instructions
        guard.protect("항상 반말로 대답해")
        guard.protect("코드 블록에 언어 태그 필수")

        # 2. Agent loop
        queries = [
            "파이썬으로 피보나치 함수 짜줘",
            "오늘 뭐 할까?",
        ]

        history: list[dict] = [{"role": "system", "content": guard.reminder()}]

        for query in queries:
            # Inject protected context as a system message for each turn.
            context = guard.context(query, token_budget=2000)
            messages = history + [
                {"role": "system", "content": context},
                {"role": "user", "content": query},
            ]

            # Call LLM
            response = chat(messages)
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})

            # 3. Check compliance
            result = guard.check(query=query, response=response)
            status_icon = "✅" if result["passed"] else "❌" if result["status"] == "failed" else "⚠️"

            print(f"\nUSER: {query}")
            print(f"AGENT: {response[:120]}...")
            print(f"COMPLIANCE: {status_icon} {result['status']} (checked={result['checked']}, violations={result['violations']})")

        # 4. Final report
        report = guard.report()
        print(f"\n--- Report ---")
        print(f"Protected: {report['protected']}, Drift warnings: {report['drift_warnings']}")
        print(f"Compliance rate: {report['compliance_rate']}")


if __name__ == "__main__":
    main()
