from __future__ import annotations

import json
import os
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

from memguard import MemoryGuard


def call_openai_compatible(
    *,
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    endpoint: str,
) -> str:
    request = Request(
        endpoint,
        data=json.dumps(
            {
                "model": model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        ).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=30) as response_obj:
        payload = json.loads(response_obj.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"].strip()


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required for this demo.")

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    endpoint = os.getenv("OPENAI_CHAT_ENDPOINT", f"{base_url}/chat/completions")

    with TemporaryDirectory(prefix="memguard-openai-demo-") as tempdir:
        guard = MemoryGuard(agent_id="openai-compatible-demo", storage_path=tempdir)
        guard.protect("항상 반말로 대답해")
        guard.protect("코드 블록에 항상 언어 태그를 붙여")
        guard.protect("파일 수정 전에는 항상 먼저 허가를 요청해")
        guard.remember("이 저장소는 pnpm을 사용해", priority="project")

        query = "FastAPI 예제를 보여주고, README를 바로 고칠지 말지도 알려줘"
        system_prompt = "\n\n".join(
            [
                "You are a coding agent. Follow the protected instructions exactly.",
                guard.reminder(),
                guard.context(query, token_budget=900),
            ]
        )
        response = call_openai_compatible(
            model=model,
            api_key=api_key,
            system_prompt=system_prompt,
            user_prompt=query,
            endpoint=endpoint,
        )

        print("\nASSISTANT RESPONSE\n")
        print(response)

        check = guard.check(query=query, response=response)
        print("\nCOMPLIANCE CHECK\n")
        print(json.dumps(check, ensure_ascii=False, indent=2))

        action = guard.observe_action(
            query="README를 바로 수정해줘",
            action="edit_file",
            target="README.md",
            requires_approval=True,
            approval_requested=False,
            approval_granted=False,
            executed=True,
        )
        print("\nACTION CHECK\n")
        print(json.dumps(action, ensure_ascii=False, indent=2))

        print("\nINTEGRITY REPORT\n")
        print(json.dumps(guard.report(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
