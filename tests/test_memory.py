from __future__ import annotations

import json

from memguard import Memory


def test_core_instruction_is_protected(tmp_path):
    mem = Memory(agent_id="assistant", storage_path=str(tmp_path))
    mem.remember("항상 반말로 대답해")
    mem.remember("오늘은 회의가 있었다")

    context = mem.build_context("오늘 날씨 어때?")

    protected_contents = [item["content"] for item in context["zones"]["protected"]]
    assert "항상 반말로 대답해" in protected_contents


def test_verify_detects_conflicting_speech_level(tmp_path):
    mem = Memory(agent_id="assistant", storage_path=str(tmp_path))
    mem.remember("항상 반말로 대답해")
    mem.remember("항상 존댓말로 대답해")

    report = mem.verify()

    assert report["integrity"]["conflicts_detected"] == 1
    assert report["conflicts"][0]["topic"] == "speech_level"


def test_verify_accepts_runtime_token_budget(tmp_path):
    mem = Memory(agent_id="assistant", storage_path=str(tmp_path))
    mem.remember("코드 블록에 항상 언어 태그를 붙여")
    mem.remember("항상 반말로 대답해")

    report = mem.verify(token_budget=10)

    assert report["protected_memories"] == 2
    assert report["integrity"]["protected_loaded"] == 1


def test_project_memory_is_recalled_for_matching_query(tmp_path):
    mem = Memory(agent_id="assistant", storage_path=str(tmp_path))
    mem.remember("이 저장소는 pnpm을 사용해", priority="project", metadata={"repo": "acme/web-app"})
    mem.remember("날씨가 맑다", priority="episode")

    result = mem.recall("web-app에서 패키지 설치 명령 알려줘", top_k=1, include_protected=False)

    assert result[0]["content"] == "이 저장소는 pnpm을 사용해"


def test_export_and_consolidate(tmp_path):
    mem = Memory(agent_id="assistant", storage_path=str(tmp_path))
    mem.remember("코드 블록에 항상 언어 태그를 붙여")
    mem.remember("코드 블록에 항상 언어 태그를 붙여")

    consolidation = mem.consolidate()
    export_path = tmp_path / "memories.json"
    mem.export(str(export_path))
    payload = json.loads(export_path.read_text(encoding="utf-8"))

    assert consolidation["merged_duplicates"] == 1
    assert len(payload) == 2


def test_replay_checks_code_block_language_tag(tmp_path):
    mem = Memory(agent_id="assistant", storage_path=str(tmp_path))
    memory_id = mem.remember("코드 블록에 항상 언어 태그를 붙여")

    failed = mem.replay(
        memory_id=memory_id,
        test_input="파이썬 예제 보여줘",
        candidate_response="```\nprint('hi')\n```",
    )
    passed = mem.replay(
        memory_id=memory_id,
        test_input="파이썬 예제 보여줘",
        candidate_response="```python\nprint('hi')\n```",
    )

    assert failed["passed"] is False
    assert "language_tag_missing" in failed["violations"]
    assert passed["passed"] is True


def test_detect_drift_flags_declining_compliance(tmp_path):
    mem = Memory(agent_id="assistant", storage_path=str(tmp_path))
    memory_id = mem.remember("항상 코드 블록에 언어 태그를 붙여")

    mem.replay(
        memory_id=memory_id,
        test_input="예제 1",
        candidate_response="```python\nprint('ok')\n```",
    )
    mem.replay(
        memory_id=memory_id,
        test_input="예제 2",
        candidate_response="```python\nprint('ok')\n```",
    )
    mem.replay(
        memory_id=memory_id,
        test_input="예제 3",
        candidate_response="```\nprint('bad')\n```",
    )
    mem.replay(
        memory_id=memory_id,
        test_input="예제 4",
        candidate_response="```\nprint('bad')\n```",
    )

    drift = mem.detect_drift()

    assert drift["inspected_memories"] == 1
    assert len(drift["drifting"]) == 1
    assert drift["drifting"][0]["instruction"] == "항상 코드 블록에 언어 태그를 붙여"


def test_observe_response_records_loaded_protected_memories(tmp_path):
    mem = Memory(agent_id="assistant", storage_path=str(tmp_path))
    mem.remember("항상 반말로 대답해")
    mem.remember("이 저장소는 pnpm을 사용해", priority="project", metadata={"repo": "acme/web-app"})

    observed = mem.observe_response(
        query="오늘 뭐 할까?",
        response="좋아, 바로 해볼게.",
    )

    assert observed["checked_memories"] == 1
    assert observed["skipped_memories"] == 0
    assert observed["passed"] == 1
    assert observed["observations"][0]["zone"] == "protected"
    assert observed["observations"][0]["source"] == "observe"


def test_observe_response_accepts_runtime_token_budget(tmp_path):
    mem = Memory(agent_id="assistant", storage_path=str(tmp_path))
    mem.remember("코드 블록에 항상 언어 태그를 붙여")
    mem.remember("항상 반말로 대답해")

    observed = mem.observe_response(
        query="파이썬 예제 보여줘",
        response="```\nprint('oops')\n```",
        token_budget=10,
    )

    assert observed["checked_memories"] == 1
    skipped_instructions = {item["instruction"] for item in observed["skipped"]}
    assert "항상 반말로 대답해" not in skipped_instructions


def test_observe_response_can_include_active_memories(tmp_path):
    mem = Memory(agent_id="assistant", storage_path=str(tmp_path))
    mem.remember("항상 반말로 대답해")
    mem.remember("이 저장소는 pnpm을 사용해", priority="project", metadata={"repo": "acme/web-app"})

    observed = mem.observe_response(
        query="web-app 패키지 설치 명령 알려줘",
        response="pnpm install 쓰면 돼.",
        include_active=True,
    )

    assert observed["checked_memories"] == 1
    assert observed["skipped_memories"] == 1
    observed_zones = {item["zone"] for item in observed["observations"]}
    skipped_zones = {item["zone"] for item in observed["skipped"]}
    assert "protected" in observed_zones
    assert "active" in skipped_zones
