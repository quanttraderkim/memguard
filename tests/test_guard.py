from __future__ import annotations

from instructionguard import MemoryGuard


def test_protect_and_check_pass(tmp_path):
    """protect 등록 → 준수하는 응답 → passed=True"""
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))
    guard.protect("항상 반말로 대답해")

    result = guard.check(
        query="오늘 뭐 할까?",
        response="좋아, 바로 해볼게.",
    )
    assert result["passed"] is True
    assert result["status"] == "passed"
    assert result["violations"] == []


def test_protect_and_check_fail(tmp_path):
    """protect 등록 → 위반하는 응답 → passed=False + violations"""
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))
    guard.protect("항상 반말로 대답해")

    result = guard.check(
        query="오늘 뭐 할까?",
        response="좋습니다. 일정을 확인해보겠습니다.",
    )
    assert result["passed"] is False
    assert result["status"] == "failed"
    assert len(result["violations"]) > 0


def test_check_skips_unrelated_protected_rules(tmp_path):
    """대화 응답 검사 시 코드 규칙 같은 무관한 checker는 skip"""
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))
    guard.protect("항상 반말로 대답해")
    guard.protect("코드 블록에 항상 언어 태그를 붙여")

    result = guard.check(
        query="오늘 뭐 할까?",
        response="좋아, 바로 해볼게.",
    )

    assert result["passed"] is True
    assert result["checked"] == 1
    assert result["skipped"] == 1
    assert result["violations"] == []
    assert result["unverified"] == []
    assert result["not_applicable"] == ["코드 블록에 항상 언어 태그를 붙여"]


def test_check_marks_unsupported_instruction_unverified(tmp_path):
    """룰 기반 체커가 없는 instruction은 false pass 대신 unverified"""
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))
    guard.protect("항상 한국어로 답해")

    result = guard.check(
        query="오늘 뭐 할까?",
        response="Sure, let's do it.",
    )

    assert result["passed"] is False
    assert result["status"] == "unverified"
    assert result["checked"] == 0
    assert result["skipped"] == 1
    assert result["unverified"] == ["항상 한국어로 답해"]
    assert result["not_applicable"] == []


def test_reminder_includes_all_protected(tmp_path):
    """reminder()가 모든 보호 지시를 포함"""
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))
    guard.protect("항상 반말로 대답해")
    guard.protect("코드 블록에 항상 언어 태그를 붙여")

    reminder = guard.reminder()
    assert "반말" in reminder
    assert "언어 태그" in reminder
    assert "⚠️" in reminder


def test_report_shows_drift(tmp_path):
    """여러 check 후 report()에 drift_warnings 표시 가능"""
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))
    guard.protect("코드 블록에 항상 언어 태그를 붙여")

    # 초반 pass
    guard.check(query="예제1", response="```python\nprint('ok')\n```")
    guard.check(query="예제2", response="```python\nprint('ok')\n```")
    # 후반 fail
    guard.check(query="예제3", response="```\nprint('bad')\n```")
    guard.check(query="예제4", response="```\nprint('bad')\n```")

    report = guard.report()
    assert "drift_warnings" in report
    assert report["protected"] >= 1
    assert report["drift_warnings"] >= 1, "drift should be detected after compliance drop"


def test_report_uses_observed_checks_for_compliance_rate(tmp_path):
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))
    guard.protect("항상 반말로 대답해")

    guard.check(query="오늘 뭐 해?", response="좋아, 바로 하자.")
    guard.check(query="오늘 뭐 해?", response="좋습니다. 진행하겠습니다.")

    report = guard.report()

    assert report["observed_checks"] == 2
    assert report["compliance_rate"] == 0.5


def test_check_accepts_runtime_token_budget(tmp_path):
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))
    guard.protect("코드 블록에 항상 언어 태그를 붙여")
    guard.protect("항상 반말로 대답해")

    large = guard.check(
        query="파이썬 예제 보여줘",
        response="```\nprint('oops')\n```",
        token_budget=4000,
    )
    small = guard.check(
        query="파이썬 예제 보여줘",
        response="```\nprint('oops')\n```",
        token_budget=10,
    )

    assert large["status"] == "failed"
    assert "항상 반말로 대답해" in large["not_applicable"]
    assert large["protected_overflow"] is False
    assert small["status"] == "failed"
    assert "항상 반말로 대답해" not in small["not_applicable"]
    assert small["protected_overflow"] is True
    assert small["omitted_protected_instructions"] == ["항상 반말로 대답해"]


def test_report_accepts_runtime_token_budget(tmp_path):
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))
    guard.protect("코드 블록에 항상 언어 태그를 붙여")
    guard.protect("항상 반말로 대답해")

    report = guard.report(token_budget=10)

    assert report["protected"] == 2
    assert report["details"]["integrity"]["protected_loaded"] == 1
    assert report["protected_overflow"] is True
    assert report["protected_omitted"] == 1


def test_context_returns_prompt_string(tmp_path):
    """context()가 문자열 반환, Protected Zone 포함"""
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))
    guard.protect("항상 반말로 대답해")

    prompt = guard.context("오늘 날씨 어때?")
    assert isinstance(prompt, str)
    assert "반말" in prompt
    assert "Protected" in prompt


def test_guard_can_set_default_token_budget(tmp_path):
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path), default_token_budget=10)
    guard.protect("코드 블록에 항상 언어 태그를 붙여")
    guard.protect("항상 반말로 대답해")

    result = guard.check(
        query="파이썬 예제 보여줘",
        response="```\nprint('oops')\n```",
    )

    assert result["status"] == "failed"
    assert "항상 반말로 대답해" not in result["not_applicable"]
    assert result["protected_overflow"] is True


def test_guard_delegates_to_memory(tmp_path):
    """remember/recall/forget이 내부 Memory로 위임"""
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))

    mid = guard.remember("테스트 기억")
    assert isinstance(mid, str)

    results = guard.recall("테스트")
    assert len(results) > 0

    guard.forget(mid)
    results = guard.recall("테스트")
    assert len(results) == 0


def test_custom_checker_can_enforce_korean_language(tmp_path):
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))

    guard.register_checker(
        "korean_language",
        evaluate=lambda **kwargs: {
            "passed": any("\uac00" <= ch <= "\ud7a3" for ch in kwargs["response"]),
            "score": 1.0 if any("\uac00" <= ch <= "\ud7a3" for ch in kwargs["response"]) else 0.0,
            "violations": [] if any("\uac00" <= ch <= "\ud7a3" for ch in kwargs["response"]) else ["korean_missing"],
        },
    )
    guard.protect("항상 한국어로 답해", checker="korean_language")

    result = guard.check(query="오늘 뭐 할까?", response="Sure, let's do it.")

    assert result["passed"] is False
    assert result["status"] == "failed"
    assert result["violations"] == ["korean_missing"]


def test_custom_checker_can_enforce_approval_request(tmp_path):
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))

    def applies_to(**kwargs):
        query = kwargs["query"]
        return "수정" in query or "edit" in query.casefold()

    def evaluate(**kwargs):
        response = kwargs["response"]
        approved = any(token in response for token in ("허가", "승인", "permission", "approve"))
        return {
            "passed": approved,
            "score": 1.0 if approved else 0.0,
            "violations": [] if approved else ["approval_request_missing"],
        }

    guard.register_checker(
        "approval_before_edit",
        applies_to=applies_to,
        evaluate=evaluate,
    )
    guard.protect("파일 수정 전에는 항상 먼저 허가를 요청해", checker="approval_before_edit")

    failed = guard.check(query="README 수정해줘", response="README.md를 바로 수정했습니다.")
    passed = guard.check(query="README 수정해줘", response="먼저 허가를 받을게. 수정해도 될까?")

    assert failed["passed"] is False
    assert failed["status"] == "failed"
    assert failed["violations"] == ["approval_request_missing"]
    assert passed["passed"] is True
    assert passed["status"] == "passed"


def test_observe_action_fails_when_edit_executes_without_approval(tmp_path):
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))
    guard.protect("파일 수정 전에는 항상 먼저 허가를 요청해")

    result = guard.observe_action(
        query="README 수정해줘",
        action="edit_file",
        target="README.md",
        requires_approval=True,
        approval_requested=False,
        approval_granted=False,
        executed=True,
    )

    assert result["passed"] is False
    assert result["status"] == "failed"
    assert "approval_request_missing" in result["violations"]
    assert "executed_without_approval" in result["violations"]


def test_observe_action_passes_when_edit_waits_for_approval(tmp_path):
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))
    guard.protect("파일 수정 전에는 항상 먼저 허가를 요청해")

    result = guard.observe_action(
        query="README 수정해줘",
        action="edit_file",
        target="README.md",
        requires_approval=True,
        approval_requested=True,
        approval_granted=False,
        executed=False,
    )

    assert result["passed"] is True
    assert result["status"] == "passed"
    assert result["violations"] == []


def test_observe_action_catches_destructive_git_command(tmp_path):
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))
    guard.protect("git reset --hard 절대 금지", kind="guardrail")

    result = guard.observe_action(
        query="브랜치 정리해줘",
        action="git_command",
        command="git reset --hard",
        executed=True,
    )

    assert result["passed"] is False
    assert result["status"] == "failed"
    assert any("git reset --hard" in violation for violation in result["violations"])


def test_observe_action_skips_git_guard_for_non_git_action(tmp_path):
    guard = MemoryGuard(agent_id="test", storage_path=str(tmp_path))
    guard.protect("git reset --hard 절대 금지", kind="guardrail")

    result = guard.observe_action(
        query="README 수정해줘",
        action="edit_file",
        target="README.md",
        executed=True,
    )

    assert result["passed"] is True
    assert result["status"] == "not_applicable"
    assert result["checked"] == 0
    assert result["skipped"] == 1
    assert result["unverified"] == []
    assert result["not_applicable"] == ["git reset --hard 절대 금지"]
