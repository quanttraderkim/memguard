from __future__ import annotations

import pytest

from instructionguard import MemoryGuard


def test_register_llm_checker_can_validate_open_ended_instruction(tmp_path):
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))

    def fake_judge(**kwargs):
        response = kwargs["response"]
        passed = any("\uac00" <= ch <= "\ud7a3" for ch in response)
        return {
            "passed": passed,
            "score": 1.0 if passed else 0.0,
            "violations": [] if passed else ["llm_semantic_mismatch"],
        }

    guard.register_llm_checker(
        "semantic_korean",
        rubric="Judge whether the response is written in Korean.",
        judge=fake_judge,
    )
    guard.protect("항상 한국어로 답해", checker="semantic_korean")

    failed = guard.check(query="오늘 뭐 해?", response="Sure, let's do it.")
    passed = guard.check(query="오늘 뭐 해?", response="좋아, 바로 하자.")

    assert failed["status"] == "failed"
    assert "korean_missing" in failed["violations"]
    assert passed["status"] == "passed"


def test_register_llm_checker_requires_model_or_custom_judge(tmp_path):
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))

    with pytest.raises(ValueError):
        guard.register_llm_checker(
            "semantic_open_ended",
            rubric="Judge semantic compliance.",
        )


def test_register_llm_checker_requires_provider_key_when_using_anthropic(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path), llm="claude-3-5-sonnet-latest")

    with pytest.raises(ValueError):
        guard.register_llm_checker(
            "semantic_claude",
            provider="anthropic",
            rubric="Judge semantic compliance.",
        )


def test_register_llm_checker_requires_provider_key_when_using_gemini(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path), llm="gemini-2.5-flash")

    with pytest.raises(ValueError):
        guard.register_llm_checker(
            "semantic_gemini",
            provider="gemini",
            rubric="Judge semantic compliance.",
        )


def test_llm_result_normalization_handles_string_booleans(tmp_path):
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))

    def fake_judge(**kwargs):
        return {
            "passed": "false",
            "score": "0.0",
            "violations": "llm_semantic_mismatch",
        }

    guard.register_llm_checker(
        "semantic_string_result",
        judge=fake_judge,
        rubric="Judge semantic compliance.",
        model="gpt-4.1-mini",
    )
    guard.protect("항상 한국어로 답해", checker="semantic_string_result")

    result = guard.check(query="오늘 뭐 해?", response="Sure, let's do it.")

    assert result["status"] == "failed"
    assert "korean_missing" in result["violations"]


def test_register_llm_checker_applies_rubric_template(tmp_path):
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))
    captured = {}

    def fake_judge(**kwargs):
        captured["rubric"] = kwargs["rubric"]
        return {
            "passed": True,
            "score": 1.0,
            "violations": [],
        }

    guard.register_llm_checker(
        "semantic_template_capture",
        judge=fake_judge,
        rubric_template="language_compliance",
        model="gpt-4.1-mini",
    )
    guard.protect("항상 한국어로 답해", checker="semantic_template_capture")

    result = guard.check(query="오늘 뭐 해?", response="좋아, 바로 하자.")

    assert result["status"] == "passed"
    assert "Ignore code blocks" in captured["rubric"]
    assert "항상 한국어로 답해" in captured["rubric"]


def test_llm_checker_conflict_with_local_signal_returns_uncertain(tmp_path):
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))

    def fake_judge(**kwargs):
        return {
            "status": "failed",
            "passed": False,
            "score": 0.95,
            "confidence": 0.95,
            "violations": ["llm_semantic_mismatch"],
        }

    guard.register_llm_checker(
        "semantic_conflict",
        judge=fake_judge,
        rubric_template="language_compliance",
        model="gpt-4.1-mini",
    )
    guard.protect("항상 한국어로 답해", checker="semantic_conflict")

    result = guard.check(query="오늘 뭐 해?", response="좋아, 바로 하자.")

    assert result["status"] == "uncertain"
    assert "llm_local_conflict" in result["violations"]
    assert result["details"][0]["resolution"] == "llm_local_conflict"


def test_llm_checker_uses_local_fallback_when_judge_is_uncertain(tmp_path):
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))

    def fake_judge(**kwargs):
        return {
            "status": "uncertain",
            "passed": False,
            "score": 0.5,
            "confidence": 0.2,
            "violations": ["llm_uncertain"],
        }

    guard.register_llm_checker(
        "semantic_summary_uncertain",
        judge=fake_judge,
        rubric_template="summary_first",
        model="gpt-4.1-mini",
    )
    guard.protect("답변 전에 한 줄 요약을 먼저 써", checker="semantic_summary_uncertain")

    result = guard.check(
        query="이 PR 요약해줘",
        response="요약: 인증 버그를 수정한다.\n자세한 내용은...",
    )

    assert result["status"] == "passed"
    assert result["details"][0]["resolution"] == "local_fallback"


def test_uncertain_llm_checks_are_reported_separately(tmp_path):
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))

    def fake_judge(**kwargs):
        return {
            "status": "failed",
            "passed": False,
            "score": 0.95,
            "confidence": 0.95,
            "violations": ["llm_semantic_mismatch"],
        }

    guard.register_llm_checker(
        "semantic_report_uncertain",
        judge=fake_judge,
        rubric_template="language_compliance",
        model="gpt-4.1-mini",
    )
    guard.protect("항상 한국어로 답해", checker="semantic_report_uncertain")

    result = guard.check(query="오늘 뭐 해?", response="좋아, 바로 하자.")
    report = guard.report()

    assert result["status"] == "uncertain"
    assert report["observed_checks"] == 1
    assert report["observed_uncertain"] == 1
    assert report["compliance_rate"] is None


def test_llm_negative_recheck_conflict_returns_uncertain(tmp_path):
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))
    calls = {"count": 0}

    def fake_judge(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "status": "failed",
                "passed": False,
                "score": 0.95,
                "confidence": 0.95,
                "violations": ["llm_semantic_mismatch"],
            }
        return {
            "status": "passed",
            "passed": True,
            "score": 0.95,
            "confidence": 0.95,
            "violations": [],
        }

    guard.register_llm_checker(
        "semantic_negative_recheck_conflict",
        judge=fake_judge,
        rubric_template="summary_first",
        negative_recheck=True,
        local_fallback=False,
        model="gpt-4.1-mini",
    )
    guard.protect("답변 전에 한 줄 요약을 먼저 써", checker="semantic_negative_recheck_conflict")

    result = guard.check(query="이 PR 요약해줘", response="세부 내용은 다음과 같아.")

    assert calls["count"] == 2
    assert result["status"] == "uncertain"
    assert result["details"][0]["resolution"] == "llm_negative_recheck_conflict"


def test_llm_negative_recheck_can_confirm_failure(tmp_path):
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))
    calls = {"count": 0}

    def fake_judge(**kwargs):
        calls["count"] += 1
        return {
            "status": "failed",
            "passed": False,
            "score": 0.9,
            "confidence": 0.9,
            "violations": ["llm_semantic_mismatch"],
        }

    guard.register_llm_checker(
        "semantic_negative_recheck_confirmed",
        judge=fake_judge,
        rubric_template="summary_first",
        negative_recheck=True,
        local_fallback=False,
        model="gpt-4.1-mini",
    )
    guard.protect("답변 전에 한 줄 요약을 먼저 써", checker="semantic_negative_recheck_confirmed")

    result = guard.check(query="이 PR 요약해줘", response="세부 내용은 다음과 같아.")

    assert calls["count"] == 2
    assert result["status"] == "failed"
    assert result["details"][0]["resolution"] == "llm_negative_recheck_confirmed"
