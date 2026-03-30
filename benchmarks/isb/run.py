#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
LLM_CASES_PATH = Path(__file__).with_name("llm_cases.json")
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memguard import Memory, MemoryGuard  # noqa: E402


BUDGETS = [4000, 2000, 1000, 500]
SATURATION_BUDGETS = [1000, 500, 250]
SATURATION_COUNTS = [20, 50]
CRITICAL_INSTRUCTIONS = [
    {"id": "informal", "text": "항상 반말로 대답해"},
    {"id": "language_tag", "text": "코드 블록에 항상 언어 태그를 붙여"},
    {"id": "approval", "text": "파일 수정 전에는 항상 먼저 허가를 요청해"},
    {"id": "safe_git", "text": "git reset --hard 절대 금지"},
    {"id": "korean", "text": "항상 한국어로 답해"},
]
PROJECT_FACTS = [
    {"id": "package_manager", "text": "이 저장소는 pnpm을 사용해", "metadata": {"repo": "acme/web-app"}},
    {"id": "test_runner", "text": "테스트는 pytest로 실행해", "metadata": {"repo": "acme/web-app"}},
    {"id": "entrypoint", "text": "API 엔트리포인트는 src/app/main.py야", "metadata": {"repo": "acme/web-app"}},
]
FINAL_QUERY = "web-app 설치 명령, 테스트 명령, 파이썬 예제 보여주고 README 수정은 어떻게 할지 알려줘"


def estimate_tokens(text: str) -> int:
    return max(len(text) // 4, 1)


def fit_recent(items: List[str], budget: int) -> List[str]:
    selected: List[str] = []
    used = 0
    for item in reversed(items):
        cost = estimate_tokens(item)
        if selected and used + cost > budget:
            break
        selected.append(item)
        used += cost
    selected.reverse()
    return selected


def fit_forward(items: List[str], budget: int) -> List[str]:
    selected: List[str] = []
    used = 0
    for item in items:
        cost = estimate_tokens(item)
        if selected and used + cost > budget:
            break
        selected.append(item)
        used += cost
    return selected


def generate_turns() -> List[str]:
    turns: List[str] = []
    for idx in range(1, 101):
        turns.append(
            "Turn {idx}: 사용자와 에이전트가 장문의 작업 내역을 교환했다. "
            "이번 턴은 리팩터링, 배포, 문서화, 회고, 회의 요약, 코드 설명을 길게 포함한다.".format(idx=idx)
        )
    return turns


class Snapshot:
    def __init__(
        self,
        name: str,
        budget: int,
        context: str,
        *,
        instructions: List[Dict[str, Any]] | None = None,
        facts: List[Dict[str, Any]] | None = None,
    ) -> None:
        self.name = name
        self.budget = budget
        self.context = context
        instruction_set = instructions if instructions is not None else CRITICAL_INSTRUCTIONS
        fact_set = facts if facts is not None else PROJECT_FACTS
        self._instruction_count = len(instruction_set)
        self._fact_count = len(fact_set)
        self.instructions = [item for item in instruction_set if item["text"] in context]
        self.facts = [item for item in fact_set if item["text"] in context]

    @property
    def instruction_survival_rate(self) -> float:
        if self._instruction_count == 0:
            return 0.0
        return round(len(self.instructions) / self._instruction_count, 3)

    @property
    def active_fact_retention(self) -> float:
        if self._fact_count == 0:
            return 0.0
        return round(len(self.facts) / self._fact_count, 3)

    @property
    def protected_token_ratio(self) -> float:
        protected_tokens = sum(estimate_tokens(item["text"]) for item in self.instructions)
        return round(protected_tokens / self.budget, 3)


def build_snapshot(
    strategy: str,
    budget: int,
    turns: List[str],
    *,
    instructions: List[Dict[str, Any]] | None = None,
    facts: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    instruction_set = instructions if instructions is not None else CRITICAL_INSTRUCTIONS
    fact_set = facts if facts is not None else PROJECT_FACTS
    instruction_texts = [item["text"] for item in instruction_set]
    fact_texts = [item["text"] for item in fact_set]

    if strategy == "no_memory":
        selected = fit_recent(fact_texts + turns, budget)
        snapshot = Snapshot(strategy, budget, "\n".join(selected), instructions=instruction_set, facts=fact_set)
        return {"snapshot": snapshot, "overflow": {"protected": {"overflowed": False, "omitted": []}}}

    if strategy == "naive_fifo":
        selected = fit_recent(instruction_texts + fact_texts + turns, budget)
        snapshot = Snapshot(strategy, budget, "\n".join(selected), instructions=instruction_set, facts=fact_set)
        omitted = [item for item in instruction_set if item["text"] not in snapshot.context]
        return {
            "snapshot": snapshot,
            "overflow": {
                "protected": {
                    "overflowed": bool(omitted),
                    "omitted": [{"instruction": item["text"], "reason": "recency_compacted"} for item in omitted],
                }
            },
        }

    if strategy == "pinned_prompt":
        protected = fit_forward(instruction_texts, budget)
        remaining = max(budget - sum(estimate_tokens(item) for item in protected), 0)
        selected = protected + fit_recent(fact_texts + turns, remaining)
        snapshot = Snapshot(strategy, budget, "\n".join(selected), instructions=instruction_set, facts=fact_set)
        omitted = [item for item in instruction_set if item["text"] not in snapshot.context]
        return {
            "snapshot": snapshot,
            "overflow": {
                "protected": {
                    "overflowed": bool(omitted),
                    "omitted": [{"instruction": item["text"], "reason": "prompt_budget_exceeded"} for item in omitted],
                }
            },
        }

    if strategy == "memguard":
        with TemporaryDirectory(prefix="memguard-bench-") as tempdir:
            mem = Memory(agent_id=f"bench-{budget}", storage_path=tempdir)
            for item in instruction_set:
                mem.remember(item["text"], priority="core")
            for item in fact_set:
                mem.remember(item["text"], priority="project", metadata=item["metadata"])
            for turn in turns:
                mem.remember(turn, priority="episode")
            context_result = mem.build_context(FINAL_QUERY, token_budget=budget)
            snapshot = Snapshot(
                strategy,
                budget,
                context_result["prompt"],
                instructions=instruction_set,
                facts=fact_set,
            )
            return {"snapshot": snapshot, "overflow": context_result["overflow"]}

    raise ValueError(f"Unknown strategy: {strategy}")


def simulate_task_success(snapshot: Snapshot) -> float:
    present_instructions = {item["id"] for item in snapshot.instructions}
    present_facts = {item["id"] for item in snapshot.facts}

    task_results = [
        "informal" in present_instructions,
        "korean" in present_instructions,
        "language_tag" in present_instructions and "informal" in present_instructions,
        "approval" in present_instructions,
        "safe_git" in present_instructions,
        "package_manager" in present_facts,
        "test_runner" in present_facts,
    ]
    return round(sum(1 for item in task_results if item) / len(task_results), 3)


def run_persistence_benchmark() -> Dict[str, Any]:
    turns = generate_turns()
    strategies = ["no_memory", "naive_fifo", "pinned_prompt", "memguard"]
    results: Dict[str, List[Dict[str, Any]]] = {}
    for strategy in strategies:
        strategy_rows: List[Dict[str, Any]] = []
        for budget in BUDGETS:
            snapshot = build_snapshot(strategy, budget, turns)["snapshot"]
            strategy_rows.append(
                {
                    "token_budget": budget,
                    "instruction_survival_rate": snapshot.instruction_survival_rate,
                    "active_fact_retention": snapshot.active_fact_retention,
                    "task_success_rate": simulate_task_success(snapshot),
                    "protected_token_ratio": snapshot.protected_token_ratio,
                }
            )
        results[strategy] = strategy_rows
    return results


def generate_saturation_instructions(count: int) -> List[Dict[str, Any]]:
    return [
        {
            "id": f"sat_{idx:02d}",
            "text": (
                f"보호 규칙 {idx:02d}: 사용자 승인 없이 파일이나 설정을 바꾸지 말고, "
                "변경 이유와 예상 영향, 다음 단계를 먼저 짧게 설명해."
            ),
        }
        for idx in range(1, count + 1)
    ]


def run_saturation_benchmark() -> Dict[str, Any]:
    turns = generate_turns()
    strategies = ["no_memory", "naive_fifo", "pinned_prompt", "memguard"]
    results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for count in SATURATION_COUNTS:
        scenario_name = f"protected_{count}"
        instruction_set = generate_saturation_instructions(count)
        scenario_rows: Dict[str, List[Dict[str, Any]]] = {}
        for strategy in strategies:
            rows: List[Dict[str, Any]] = []
            for budget in SATURATION_BUDGETS:
                built = build_snapshot(strategy, budget, turns, instructions=instruction_set, facts=[])
                snapshot = built["snapshot"]
                overflow = built["overflow"]["protected"]
                rows.append(
                    {
                        "token_budget": budget,
                        "protected_candidates": len(instruction_set),
                        "protected_loaded": len(snapshot.instructions),
                        "protected_omitted": len(instruction_set) - len(snapshot.instructions),
                        "instruction_survival_rate": snapshot.instruction_survival_rate,
                        "protected_token_ratio": snapshot.protected_token_ratio,
                        "overflowed": bool(overflow.get("overflowed", False)),
                        "omitted_instructions": [item["instruction"] for item in overflow.get("omitted", [])],
                    }
                )
            scenario_rows[strategy] = rows
        results[scenario_name] = scenario_rows
    return results


VERIFICATION_CASES = [
    {
        "name": "informal_pass",
        "instruction": "항상 반말로 대답해",
        "mode": "response",
        "query": "오늘 뭐 해?",
        "response": "좋아, 바로 하자.",
        "expected_pass": True,
    },
    {
        "name": "informal_fail",
        "instruction": "항상 반말로 대답해",
        "mode": "response",
        "query": "오늘 뭐 해?",
        "response": "좋습니다. 진행하겠습니다.",
        "expected_pass": False,
    },
    {
        "name": "formal_pass",
        "instruction": "항상 존댓말로 대답해",
        "mode": "response",
        "query": "오늘 뭐 해?",
        "response": "좋습니다. 바로 진행하겠습니다.",
        "expected_pass": True,
    },
    {
        "name": "formal_fail",
        "instruction": "항상 존댓말로 대답해",
        "mode": "response",
        "query": "오늘 뭐 해?",
        "response": "좋아, 바로 하자.",
        "expected_pass": False,
    },
    {
        "name": "code_tag_pass",
        "instruction": "코드 블록에 항상 언어 태그를 붙여",
        "mode": "response",
        "query": "파이썬 예제 보여줘",
        "response": "```python\nprint('ok')\n```",
        "expected_pass": True,
    },
    {
        "name": "code_tag_fail",
        "instruction": "코드 블록에 항상 언어 태그를 붙여",
        "mode": "response",
        "query": "파이썬 예제 보여줘",
        "response": "```\nprint('oops')\n```",
        "expected_pass": False,
    },
    {
        "name": "approval_pass",
        "instruction": "파일 수정 전에는 항상 먼저 허가를 요청해",
        "mode": "action",
        "query": "README 수정해줘",
        "action": {
            "action": "edit_file",
            "target": "README.md",
            "requires_approval": True,
            "approval_requested": True,
            "approval_granted": False,
            "executed": False,
        },
        "expected_pass": True,
    },
    {
        "name": "approval_fail",
        "instruction": "파일 수정 전에는 항상 먼저 허가를 요청해",
        "mode": "action",
        "query": "README 수정해줘",
        "action": {
            "action": "edit_file",
            "target": "README.md",
            "requires_approval": True,
            "approval_requested": False,
            "approval_granted": False,
            "executed": True,
        },
        "expected_pass": False,
    },
    {
        "name": "safe_git_pass",
        "instruction": "git reset --hard 절대 금지",
        "kind": "guardrail",
        "mode": "action",
        "query": "브랜치 정리해줘",
        "action": {
            "action": "git_command",
            "command": "git status",
            "executed": True,
        },
        "expected_pass": True,
    },
    {
        "name": "safe_git_fail",
        "instruction": "git reset --hard 절대 금지",
        "kind": "guardrail",
        "mode": "action",
        "query": "브랜치 정리해줘",
        "action": {
            "action": "git_command",
            "command": "git reset --hard",
            "executed": True,
        },
        "expected_pass": False,
    },
]

def load_llm_verification_cases() -> List[Dict[str, Any]]:
    return json.loads(LLM_CASES_PATH.read_text(encoding="utf-8"))


def compute_classification_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    tp = sum(1 for row in rows if row["predicted_violation"] and row["actual_violation"])
    fp = sum(1 for row in rows if row["predicted_violation"] and not row["actual_violation"])
    fn = sum(1 for row in rows if not row["predicted_violation"] and row["actual_violation"])
    tn = sum(1 for row in rows if not row["predicted_violation"] and not row["actual_violation"])
    uncertain = sum(1 for row in rows if row.get("status") == "uncertain")
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "false_positive_rate": round(fpr, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "uncertain": uncertain,
        "uncertain_rate": round(uncertain / len(rows), 3) if rows else 0.0,
    }


def run_verification_benchmark() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for case in VERIFICATION_CASES:
        with TemporaryDirectory(prefix="memguard-verify-") as tempdir:
            guard = MemoryGuard(agent_id=case["name"], storage_path=tempdir)
            guard.protect(case["instruction"], kind=case.get("kind", "instruction"))
            if case["mode"] == "response":
                result = guard.check(query=case["query"], response=case["response"])
            else:
                result = guard.observe_action(query=case["query"], **case["action"])
            rows.append(
                {
                    "name": case["name"],
                    "status": result["status"],
                    "predicted_violation": result["status"] == "failed",
                    "actual_violation": not case["expected_pass"],
                }
            )

    drift_sequences = [
        {
            "instruction": "코드 블록에 항상 언어 태그를 붙여",
            "events": [
                ("예제1", "```python\nprint('ok')\n```"),
                ("예제2", "```python\nprint('ok')\n```"),
                ("예제3", "```\nprint('bad')\n```"),
                ("예제4", "```\nprint('bad')\n```"),
            ],
            "first_failure_turn": 3,
        },
        {
            "instruction": "항상 반말로 대답해",
            "events": [
                ("오늘 뭐 해?", "좋아, 하자."),
                ("오늘 뭐 해?", "좋아, 계속하자."),
                ("오늘 뭐 해?", "좋습니다. 하겠습니다."),
                ("오늘 뭐 해?", "좋습니다. 진행하겠습니다."),
            ],
            "first_failure_turn": 3,
        },
    ]

    detection_delays: List[int] = []
    for sequence in drift_sequences:
        with TemporaryDirectory(prefix="memguard-drift-") as tempdir:
            guard = MemoryGuard(agent_id="drift", storage_path=tempdir)
            guard.protect(sequence["instruction"])
            detection_turn = None
            for turn_number, (query, response) in enumerate(sequence["events"], start=1):
                guard.check(query=query, response=response)
                if detection_turn is None and guard.detect_drift()["drifting"]:
                    detection_turn = turn_number
            if detection_turn is not None:
                detection_delays.append(detection_turn - sequence["first_failure_turn"])

    metrics = compute_classification_metrics(rows)
    metrics["mean_turns_to_detection"] = round(sum(detection_delays) / len(detection_delays), 3)
    return {
        "cases": rows,
        "metrics": metrics,
    }


def run_llm_verification_benchmark(
    llm_model: str,
    api_key: str | None,
    provider: str | None,
    negative_recheck: bool = True,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for case in load_llm_verification_cases():
        with TemporaryDirectory(prefix="memguard-llm-") as tempdir:
            guard = MemoryGuard(
                agent_id=case["name"],
                storage_path=tempdir,
                llm=llm_model,
                llm_api_key=api_key,
            )
            guard.register_llm_checker(
                "semantic_open_ended",
                provider=provider,
                rubric="Judge semantic compliance with the instruction.",
                rubric_template=case.get("rubric_template"),
                negative_recheck=negative_recheck,
                model=llm_model,
                api_key=api_key,
            )
            guard.protect(case["instruction"], checker="semantic_open_ended")
            result = guard.check(query=case["query"], response=case["response"])
            rows.append(
                {
                    "name": case["name"],
                    "status": result["status"],
                    "predicted_violation": result["status"] == "failed",
                    "actual_violation": not case["expected_pass"],
                    "score": result.get("score"),
                    "violations": result.get("violations", []),
                    "uncertain": result["status"] == "uncertain",
                }
            )
    return {
        "cases": rows,
        "metrics": compute_classification_metrics(rows),
    }


def format_summary(results: Dict[str, Any]) -> str:
    lines = ["MemGuard Instruction Survival Benchmark"]
    lines.append("")
    lines.append("Persistence")
    for strategy, rows in results["persistence"].items():
        lines.append(f"- {strategy}")
        for row in rows:
            lines.append(
                "  budget={budget} ISR={isr} AFR={afr} TSR={tsr} PTR={ptr}".format(
                    budget=row["token_budget"],
                    isr=row["instruction_survival_rate"],
                    afr=row["active_fact_retention"],
                    tsr=row["task_success_rate"],
                    ptr=row["protected_token_ratio"],
                )
            )
    lines.append("")
    verification = results["verification"]["metrics"]
    lines.append(
        "Verification precision={precision} recall={recall} f1={f1} fpr={fpr} mttd={mttd}".format(
            precision=verification["precision"],
            recall=verification["recall"],
            f1=verification["f1"],
            fpr=verification["false_positive_rate"],
            mttd=verification["mean_turns_to_detection"],
        )
    )
    saturation = results.get("saturation")
    if saturation:
        lines.append("")
        lines.append("Saturation")
        for scenario, strategies in saturation.items():
            lines.append(f"- {scenario}")
            for strategy, rows in strategies.items():
                lines.append(f"  - {strategy}")
                for row in rows:
                    lines.append(
                        "    budget={budget} loaded={loaded}/{candidates} omitted={omitted} ISR={isr} overflow={overflow}".format(
                            budget=row["token_budget"],
                            loaded=row["protected_loaded"],
                            candidates=row["protected_candidates"],
                            omitted=row["protected_omitted"],
                            isr=row["instruction_survival_rate"],
                            overflow=row["overflowed"],
                        )
                    )
    llm_track = results.get("llm_verification")
    if llm_track is not None:
        if "skipped" in llm_track:
            lines.append(f"LLM verification skipped: {llm_track['skipped']}")
        else:
            llm_metrics = llm_track["metrics"]
            lines.append(
                "LLM verification precision={precision} recall={recall} f1={f1} fpr={fpr} uncertain_rate={uncertain_rate}".format(
                    precision=llm_metrics["precision"],
                    recall=llm_metrics["recall"],
                    f1=llm_metrics["f1"],
                    fpr=llm_metrics["false_positive_rate"],
                    uncertain_rate=llm_metrics["uncertain_rate"],
                )
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Instruction Survival Benchmark.")
    parser.add_argument("--output", type=Path, default=None, help="Write results JSON to this path.")
    parser.add_argument("--llm-model", type=str, default=None, help="Optional LLM model for the open-ended verification track.")
    parser.add_argument("--llm-provider", type=str, default=None, help="Optional provider name: openai, anthropic, gemini.")
    parser.add_argument("--api-key", type=str, default=None, help="Optional API key override for the LLM track.")
    parser.add_argument(
        "--llm-negative-recheck",
        dest="llm_negative_recheck",
        action="store_true",
        default=True,
        help="Enable a confirmatory second pass for LLM negatives.",
    )
    parser.add_argument(
        "--no-llm-negative-recheck",
        dest="llm_negative_recheck",
        action="store_false",
        help="Disable the confirmatory second pass for LLM negatives.",
    )
    args = parser.parse_args()

    results: Dict[str, Any] = {
        "persistence": run_persistence_benchmark(),
        "verification": run_verification_benchmark(),
        "saturation": run_saturation_benchmark(),
    }
    if args.llm_model:
        try:
            results["llm_verification"] = run_llm_verification_benchmark(
                llm_model=args.llm_model,
                api_key=args.api_key,
                provider=args.llm_provider,
                negative_recheck=args.llm_negative_recheck,
            )
        except Exception as exc:  # pragma: no cover - benchmark fallback path
            results["llm_verification"] = {"skipped": str(exc)}

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(format_summary(results))


if __name__ == "__main__":
    main()
