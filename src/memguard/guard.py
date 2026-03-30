from __future__ import annotations

from typing import Any, Dict, List, Optional

from .core import Memory

_UNVERIFIED_REASONS = {
    "no_rule_based_checker",
    "no_rule_based_checker_for_topic",
    "unknown_checker",
}


class MemoryGuard:
    """High-level API for memory integrity protection.

    Wraps the lower-level Memory class with a simplified interface
    focused on protecting instructions and verifying compliance.
    """

    def __init__(
        self,
        agent_id: str,
        storage_path: Optional[str] = None,
        llm: Optional[str] = None,
        llm_api_key: Optional[str] = None,  # TODO: v0.3에서 LLM 어댑터에 전달 예정
        default_token_budget: int = 4000,
    ) -> None:
        self._mem = Memory(
            agent_id=agent_id,
            storage_path=storage_path,
            llm=llm,
        )
        self.agent_id = agent_id
        self._llm_api_key = llm_api_key
        self._default_token_budget = default_token_budget

    def protect(
        self,
        instruction: str,
        *,
        kind: str = "instruction",
        checker: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register a protected instruction. Returns memory_id."""
        return self._mem.remember(
            instruction,
            priority="core",
            kind=kind,
            checker=checker,
            source="guard",
            metadata=metadata,
        )

    def check(
        self,
        *,
        query: str,
        response: str,
        token_budget: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Check whether a response complies with all protected instructions."""
        result = self._mem.observe_response(
            query=query,
            response=response,
            token_budget=self._resolve_token_budget(token_budget),
            source="guard_check",
        )
        return self._summarize_observation_result(result)

    def observe_action(
        self,
        *,
        query: str,
        action: str,
        target: Optional[str] = None,
        command: Optional[str] = None,
        requires_approval: bool = False,
        approval_requested: bool = False,
        approval_granted: bool = False,
        executed: bool = True,
        include_active: bool = False,
        token_budget: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Observe an execution event and verify action-level compliance."""
        result = self._mem.observe_action(
            query=query,
            action=action,
            target=target,
            command=command,
            requires_approval=requires_approval,
            approval_requested=approval_requested,
            approval_granted=approval_granted,
            executed=executed,
            include_active=include_active,
            token_budget=self._resolve_token_budget(token_budget),
            source="guard_action",
            metadata=metadata,
        )
        summary = self._summarize_observation_result(result)
        summary["action"] = result["action"]
        return summary

    def _summarize_observation_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        all_violations: List[str] = []
        min_score = 1.0
        uncertain_count = 0
        for obs in result["observations"]:
            all_violations.extend(obs.get("violations", []))
            if obs["score"] < min_score:
                min_score = obs["score"]
            if obs.get("status") == "uncertain":
                uncertain_count += 1

        checked = result["checked_memories"]
        skipped = result.get("skipped_memories", 0)
        skipped_details = result.get("skipped", [])
        unverified_details = [
            item for item in skipped_details if item.get("reason") in _UNVERIFIED_REASONS
        ]
        not_applicable_details = [
            item for item in skipped_details if item.get("reason") not in _UNVERIFIED_REASONS
        ]
        if checked == 0 and unverified_details:
            status = "unverified"
            passed = False
            score = None
        elif checked == 0 and not_applicable_details:
            status = "not_applicable"
            passed = True
            score = None
        else:
            if result["failed"] > 0:
                status = "failed"
            elif uncertain_count > 0:
                status = "uncertain"
            else:
                status = "passed"
            passed = status == "passed"
            score = min_score if result["observations"] else 1.0

        return {
            "passed": passed,
            "status": status,
            "violations": all_violations,
            "score": score,
            "checked": checked,
            "skipped": skipped,
            "uncertain": uncertain_count,
            "unverified": [item["instruction"] for item in unverified_details],
            "not_applicable": [item["instruction"] for item in not_applicable_details],
            "details": result["observations"],
            "skipped_details": skipped_details,
        }

    def reminder(self) -> str:
        """Generate a reminder prompt containing all protected instructions."""
        protected = self._mem.get_protected()
        if not protected:
            return ""
        lines = ["⚠️ 다음 지시를 반드시 준수하세요:"]
        for item in protected:
            lines.append(f"- {item.content}")
        return "\n".join(lines)

    def report(self, *, token_budget: Optional[int] = None) -> Dict[str, Any]:
        """Generate a full integrity report."""
        verification = self._mem.verify(token_budget=self._resolve_token_budget(token_budget))
        stats = self._mem.stats()
        integrity = verification["integrity"]

        total_checks = integrity["instruction_checks_total"]
        decided_checks = integrity["instruction_checks_passed"] + integrity["instruction_checks_failed"]
        compliance_rate = (
            round(integrity["instruction_checks_passed"] / decided_checks, 3)
            if decided_checks > 0
            else None
        )

        return {
            "protected": integrity["protected_expected"],
            "active": stats["total"] - integrity["protected_expected"],
            "drift_warnings": integrity["drift_warnings"],
            "conflicts": integrity["conflicts_detected"],
            "observed_checks": total_checks,
            "observed_uncertain": integrity.get("instruction_checks_uncertain", 0),
            "compliance_rate": compliance_rate,
            "details": verification,
        }

    def context(self, query: str, *, token_budget: Optional[int] = None) -> str:
        """Build context prompt string with protected zone."""
        result = self._mem.build_context(query, token_budget=self._resolve_token_budget(token_budget))
        return result["prompt"]

    # --- Delegate to Memory ---

    def remember(self, content: str, **kwargs: Any) -> str:
        """Store a memory. See Memory.remember() for full options."""
        return self._mem.remember(content, **kwargs)

    def recall(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """Recall memories matching a query. See Memory.recall() for options."""
        return self._mem.recall(query, **kwargs)

    def forget(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        self._mem.forget(memory_id)

    def verify(self, *, token_budget: Optional[int] = None) -> Dict[str, Any]:
        """Run full integrity verification. See Memory.verify()."""
        return self._mem.verify(token_budget=self._resolve_token_budget(token_budget))

    def detect_drift(self) -> Dict[str, Any]:
        """Detect instruction compliance drift. See Memory.detect_drift()."""
        return self._mem.detect_drift()

    def export(self, path: str) -> None:
        """Export all memories to JSON file."""
        self._mem.export(path)

    def stats(self) -> Dict[str, Any]:
        """Return memory statistics."""
        return self._mem.stats()

    def register_checker(
        self,
        name: str,
        *,
        evaluate: Any,
        applies_to: Optional[Any] = None,
    ) -> None:
        """Register a custom checker for this process."""
        self._mem.register_checker(name, evaluate=evaluate, applies_to=applies_to)

    def available_checkers(self) -> List[str]:
        """Return built-in and custom checker names."""
        return self._mem.available_checkers()

    def _resolve_token_budget(self, token_budget: Optional[int]) -> int:
        return self._default_token_budget if token_budget is None else token_budget

    def register_llm_checker(
        self,
        name: str,
        *,
        provider: Optional[str] = None,
        rubric: Optional[str] = None,
        rubric_template: Optional[str] = None,
        uncertainty_threshold: float = 0.75,
        local_fallback: bool = True,
        negative_recheck: bool = False,
        applies_to: Optional[Any] = None,
        judge: Optional[Any] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> None:
        """Register an optional LLM-backed checker."""
        self._mem.register_llm_checker(
            name,
            provider=provider,
            rubric=rubric,
            rubric_template=rubric_template,
            uncertainty_threshold=uncertainty_threshold,
            local_fallback=local_fallback,
            negative_recheck=negative_recheck,
            applies_to=applies_to,
            judge=judge,
            model=model,
            api_key=api_key or self._llm_api_key,
            endpoint=endpoint,
        )
