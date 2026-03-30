from __future__ import annotations

from pprint import pprint
from tempfile import TemporaryDirectory

from instructionguard import Memory


def run_turn(mem: Memory, query: str, response: str) -> None:
    context = mem.build_context(query, token_budget=600, include_buffer=False)
    observation = mem.observe_response(
        query=query,
        response=response,
        include_active=True,
    )

    print(f"\nUSER: {query}")
    print("CONTEXT:")
    print(context["prompt"])
    print("ASSISTANT:")
    print(response)
    print("OBSERVATION:")
    pprint(
        {
            "checked_memories": observation["checked_memories"],
            "failed": observation["failed"],
            "checks": [
                {
                    "instruction": item["instruction"],
                    "passed": item["passed"],
                    "zone": item["zone"],
                    "violations": item["violations"],
                }
                for item in observation["observations"]
            ],
        }
    )


def main() -> None:
    with TemporaryDirectory(prefix="instructionguard-demo-") as tempdir:
        mem = Memory(agent_id="demo-agent", storage_path=tempdir)

        mem.remember("항상 반말로 대답해")
        mem.remember("코드 블록에 항상 언어 태그를 붙여")
        mem.remember("이 저장소는 pnpm을 사용해", priority="project", metadata={"repo": "acme/web-app"})

        turns = [
            ("web-app에서 FastAPI 예제 보여줘", "좋아, 이렇게 해봐.\n```python\nprint('hello')\n```"),
            ("다른 파이썬 예제도 보여줘", "좋아, 이것도 봐.\n```python\nprint('world')\n```"),
            ("예제 하나만 더", "좋아, 이렇게 해.\n```\nprint('oops')\n```"),
            ("마지막 예제 보여줘", "좋아, 이것도 가능해.\n```\nprint('oops again')\n```"),
        ]

        for query, response in turns:
            run_turn(mem, query, response)

        print("\nVERIFY:")
        pprint(mem.verify())

        print("\nDRIFT:")
        pprint(mem.detect_drift())


if __name__ == "__main__":
    main()
