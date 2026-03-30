from __future__ import annotations

import json
import re
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .classifier import infer_memory_profile, is_protected, lexical_score, normalize_text
from .llm import build_llm_checker
from .models import ComplianceEvent, MemoryItem, MemoryKind, MemoryPriority, MemoryStatus
from .storage import SQLiteStore

_CUSTOM_CHECKERS: Dict[str, Dict[str, Any]] = {}


class Memory:
    def __init__(
        self,
        agent_id: str,
        storage_path: Optional[str] = None,
        mode: str = "local",
        llm: Optional[str] = None,
    ) -> None:
        if mode != "local":
            raise ValueError("v0.1 supports only mode='local'")
        self.agent_id = agent_id
        self.mode = mode
        self.llm = llm
        self.store = SQLiteStore(_resolve_storage_path(agent_id, storage_path))

    def remember(
        self,
        content: str,
        *,
        priority: Optional[str] = None,
        kind: Optional[str] = None,
        checker: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if checker is not None and checker not in self.available_checkers():
            raise ValueError("Unknown checker: {name}".format(name=checker))
        final_priority, final_kind, final_metadata = infer_memory_profile(
            content,
            priority=priority,
            kind=kind,
            metadata=metadata,
        )
        if checker is not None:
            final_metadata["checker"] = checker
        now = _utcnow()
        reinforcement_score = self._calculate_reinforcement(content, final_metadata)
        persistence_score = self._calculate_persistence(final_priority, reinforcement_score)
        item = MemoryItem(
            id=str(uuid.uuid4()),
            content=content.strip(),
            priority=final_priority,
            kind=final_kind,
            status=MemoryStatus.ACTIVE,
            source=source or "manual",
            scope=str(final_metadata.get("scope", "global")),
            reinforcement_score=reinforcement_score,
            persistence_score=persistence_score,
            last_used_at=None,
            created_at=now,
            updated_at=now,
            metadata=final_metadata,
        )
        self.store.add(self.agent_id, item)
        return item.id

    def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
        include_protected: bool = True,
    ) -> List[Dict[str, Any]]:
        candidates: List[Tuple[float, MemoryItem]] = []
        for item in self._active_items():
            if not include_protected and is_protected(item.priority):
                continue
            score = lexical_score(query, item.content, item.metadata)
            score += item.persistence_score * 0.05
            if score <= 0:
                continue
            candidates.append((score, item))

        ranked = [item for _, item in sorted(candidates, key=lambda pair: pair[0], reverse=True)[:top_k]]
        self._mark_used(ranked)
        return [item.to_dict() for item in ranked]

    def build_context(
        self,
        query: str,
        *,
        token_budget: int = 4000,
        include_buffer: bool = True,
    ) -> Dict[str, Any]:
        protected = self._protected_items()
        active_candidates = [
            self.store.get(self.agent_id, entry["id"])
            for entry in self.recall(query, top_k=8, include_protected=False)
        ]
        active = [item for item in active_candidates if item is not None]
        buffer_items = self._buffer_items() if include_buffer else []

        protected_budget = max(int(token_budget * 0.5), 1)
        active_budget = max(int(token_budget * 0.35), 1)
        buffer_budget = max(token_budget - protected_budget - active_budget, 0)

        protected_fit = _fit_items_to_budget_detailed(protected, protected_budget)
        active_fit = _fit_items_to_budget_detailed(active, active_budget)
        buffer_fit = _fit_items_to_budget_detailed(buffer_items, buffer_budget)

        protected_selected = protected_fit["selected"]
        active_selected = active_fit["selected"]
        buffer_selected = buffer_fit["selected"]

        prompt = _render_context(protected_selected, active_selected, buffer_selected)
        return {
            "query": query,
            "token_budget": token_budget,
            "budgets": {
                "protected": protected_budget,
                "active": active_budget,
                "buffer": buffer_budget,
            },
            "zones": {
                "protected": [item.to_dict() for item in protected_selected],
                "active": [item.to_dict() for item in active_selected],
                "buffer": [item.to_dict() for item in buffer_selected],
            },
            "overflow": {
                "protected": _selection_overflow_summary(
                    zone="protected",
                    fit=protected_fit,
                    budget=protected_budget,
                ),
                "active": _selection_overflow_summary(
                    zone="active",
                    fit=active_fit,
                    budget=active_budget,
                ),
                "buffer": _selection_overflow_summary(
                    zone="buffer",
                    fit=buffer_fit,
                    budget=buffer_budget,
                ),
            },
            "prompt": prompt,
        }

    def verify(self, *, token_budget: int = 4000) -> Dict[str, Any]:
        protected = self._protected_items()
        context = self.build_context("", token_budget=token_budget, include_buffer=False)
        loaded_ids = {item["id"] for item in context["zones"]["protected"]}
        conflicts = _detect_conflicts(protected)
        drift = self.detect_drift()
        events = self.store.list_events(self.agent_id)
        passed_checks = sum(1 for event in events if event.details.get("status", "passed" if event.passed else "failed") == "passed")
        failed_checks = sum(1 for event in events if event.details.get("status", "passed" if event.passed else "failed") == "failed")
        uncertain_checks = sum(
            1 for event in events if event.details.get("status", "passed" if event.passed else "failed") == "uncertain"
        )

        return {
            "total_memories": len(self.store.list(self.agent_id)),
            "protected_memories": len(protected),
            "integrity": {
                "protected_loaded": len(loaded_ids),
                "protected_expected": len(protected),
                "protected_overflow": context["overflow"]["protected"]["overflowed"],
                "protected_omitted": context["overflow"]["protected"]["omitted_count"],
                "protected_budget": context["overflow"]["protected"]["budget"],
                "protected_used_tokens": context["overflow"]["protected"]["used_tokens"],
                "instruction_checks_passed": passed_checks,
                "instruction_checks_failed": failed_checks,
                "instruction_checks_uncertain": uncertain_checks,
                "instruction_checks_total": len(events),
                "conflicts_detected": len(conflicts),
                "drift_warnings": len(drift["drifting"]),
            },
            "conflicts": conflicts,
            "drift": drift,
            "overflow": context["overflow"],
        }

    def replay(
        self,
        *,
        memory_id: str,
        test_input: str,
        expected_check: Optional[str] = None,
        candidate_response: Optional[str] = None,
        source: str = "replay",
    ) -> Dict[str, Any]:
        memory = self.store.get(self.agent_id, memory_id)
        if memory is None:
            raise ValueError(f"Unknown memory_id: {memory_id}")

        context = self.build_context(test_input, include_buffer=False)
        checker = _resolve_checker(memory, expected_check)
        effective_response = candidate_response or _simulate_response(memory, checker, test_input)
        return self._record_compliance_event(
            memory=memory,
            query=test_input,
            response=effective_response,
            checker=checker,
            source=source,
            context_zones=context["zones"],
            expected_check=expected_check,
        )

    def observe_response(
        self,
        *,
        query: str,
        response: str,
        memory_ids: Optional[List[str]] = None,
        include_active: bool = False,
        token_budget: int = 4000,
        source: str = "observe",
    ) -> Dict[str, Any]:
        context = self.build_context(query, token_budget=token_budget, include_buffer=False)
        loaded_memory_map = _collect_loaded_memories(self.agent_id, self.store, context["zones"], include_active)

        if memory_ids is not None:
            requested_ids = set(memory_ids)
            loaded_memory_map = {
                memory_id: item for memory_id, item in loaded_memory_map.items() if memory_id in requested_ids
            }

        return self._observe_loaded_memories(
            query=query,
            response=response,
            loaded_memory_map=loaded_memory_map,
            source=source,
            context_zones=context["zones"],
            overflow=context["overflow"],
        )

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
        memory_ids: Optional[List[str]] = None,
        include_active: bool = False,
        token_budget: int = 4000,
        source: str = "observe_action",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = self.build_context(query, token_budget=token_budget, include_buffer=False)
        loaded_memory_map = _collect_loaded_memories(self.agent_id, self.store, context["zones"], include_active)
        if memory_ids is not None:
            requested_ids = set(memory_ids)
            loaded_memory_map = {
                memory_id: item for memory_id, item in loaded_memory_map.items() if memory_id in requested_ids
            }

        event_context = {
            "event_type": "action",
            "action": action,
            "target": target,
            "command": command,
            "requires_approval": requires_approval,
            "approval_requested": approval_requested,
            "approval_granted": approval_granted,
            "executed": executed,
            "metadata": dict(metadata or {}),
        }
        response = _render_action_event(event_context)
        observed = self._observe_loaded_memories(
            query=query,
            response=response,
            loaded_memory_map=loaded_memory_map,
            source=source,
            context_zones=context["zones"],
            overflow=context["overflow"],
            event_context=event_context,
        )
        observed["action"] = event_context
        return observed

    def detect_drift(
        self,
        *,
        min_checks: int = 4,
        min_drop: float = 0.25,
    ) -> Dict[str, Any]:
        drifting: List[Dict[str, Any]] = []
        inspected = 0

        for memory in self._protected_items():
            events = self.store.list_events(self.agent_id, memory.id)
            if len(events) < min_checks:
                continue
            inspected += 1
            scores = [round(event.score, 3) for event in events]
            midpoint = max(len(scores) // 2, 1)
            earlier_avg = sum(scores[:midpoint]) / len(scores[:midpoint])
            recent_avg = sum(scores[midpoint:]) / len(scores[midpoint:])
            drop = round(earlier_avg - recent_avg, 3)
            if drop < min_drop or recent_avg >= earlier_avg:
                continue

            drifting.append(
                {
                    "memory_id": memory.id,
                    "instruction": memory.content,
                    "checker": _resolve_checker(memory, None),
                    "compliance_trend": scores,
                    "average_before": round(earlier_avg, 3),
                    "average_recent": round(recent_avg, 3),
                    "drop": drop,
                    "status": "warning" if recent_avg > 0.4 else "critical",
                    "last_event_at": events[-1].created_at.isoformat(),
                }
            )

        return {
            "inspected_memories": inspected,
            "drifting": drifting,
        }

    def consolidate(self) -> Dict[str, Any]:
        items = self._active_items()
        by_normalized: dict[str, list[MemoryItem]] = {}
        for item in items:
            by_normalized.setdefault(normalize_text(item.content), []).append(item)

        archived = 0
        merged = 0
        for duplicates in by_normalized.values():
            if len(duplicates) < 2:
                continue
            canonical = duplicates[0]
            merged += len(duplicates) - 1
            for duplicate in duplicates[1:]:
                duplicate.status = MemoryStatus.ARCHIVED
                duplicate.updated_at = _utcnow()
                self.store.update(self.agent_id, duplicate)
                archived += 1
            canonical.reinforcement_score += len(duplicates) - 1
            canonical.persistence_score = self._calculate_persistence(
                canonical.priority,
                canonical.reinforcement_score,
            )
            canonical.updated_at = _utcnow()
            self.store.update(self.agent_id, canonical)

        return {"merged_duplicates": merged, "archived_items": archived}

    def forget(self, memory_id: str) -> None:
        self.store.delete(self.agent_id, memory_id)

    def export(self, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = [item.to_dict() for item in self.store.list(self.agent_id)]
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def stats(self) -> Dict[str, Any]:
        items = self.store.list(self.agent_id)
        by_priority: dict[str, int] = {}
        for item in items:
            by_priority[item.priority.value] = by_priority.get(item.priority.value, 0) + 1
        return {
            "total": len(items),
            "by_priority": by_priority,
            "compliance_events": len(self.store.list_events(self.agent_id)),
        }

    @classmethod
    def register_checker(
        cls,
        name: str,
        *,
        evaluate: Any,
        applies_to: Optional[Any] = None,
    ) -> None:
        _CUSTOM_CHECKERS[name] = {
            "evaluate": evaluate,
            "applies_to": applies_to,
        }

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
        evaluate = build_llm_checker(
            model=model or self.llm,
            provider=provider,
            api_key=api_key,
            endpoint=endpoint,
            rubric=rubric,
            rubric_template=rubric_template,
            uncertainty_threshold=uncertainty_threshold,
            local_fallback=local_fallback,
            negative_recheck=negative_recheck,
            judge=judge,
        )
        self.register_checker(name, evaluate=evaluate, applies_to=applies_to)

    @classmethod
    def available_checkers(cls) -> List[str]:
        builtins = [
            "korean_informal",
            "korean_formal",
            "codeblock_language_tag",
            "approval_before_action",
            "no_destructive_git_commands",
            "memory_loaded",
        ]
        return sorted(builtins + list(_CUSTOM_CHECKERS.keys()))

    def _observe_loaded_memories(
        self,
        *,
        query: str,
        response: str,
        loaded_memory_map: Dict[str, MemoryItem],
        source: str,
        context_zones: Dict[str, List[Dict[str, Any]]],
        overflow: Optional[Dict[str, Any]] = None,
        event_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        observations: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        for memory in loaded_memory_map.values():
            checker = _resolve_checker(memory, None)
            applicability = _checker_applicability(
                memory,
                checker,
                query,
                response,
                event_context=event_context,
            )
            if not applicability["applicable"]:
                skipped.append(
                    {
                        "memory_id": memory.id,
                        "instruction": memory.content,
                        "checker": checker,
                        "reason": applicability["reason"],
                        "zone": _locate_memory_zone(memory.id, context_zones),
                        "source": source,
                    }
                )
                continue
            observations.append(
                self._record_compliance_event(
                    memory=memory,
                    query=query,
                    response=response,
                    checker=checker,
                    source=source,
                    context_zones=context_zones,
                    expected_check=None,
                    event_context=event_context,
                )
            )

        passed_count = sum(1 for observation in observations if observation["status"] == "passed")
        failed_count = sum(1 for observation in observations if observation["status"] == "failed")
        uncertain_count = sum(1 for observation in observations if observation["status"] == "uncertain")

        return {
            "query": query,
            "checked_memories": len(observations),
            "skipped_memories": len(skipped),
            "passed": passed_count,
            "failed": failed_count,
            "uncertain": uncertain_count,
            "observations": observations,
            "skipped": skipped,
            "overflow": overflow or {},
        }

    def _record_compliance_event(
        self,
        *,
        memory: MemoryItem,
        query: str,
        response: str,
        checker: str,
        source: str,
        context_zones: Dict[str, List[Dict[str, Any]]],
        expected_check: Optional[str],
        event_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        zone = _locate_memory_zone(memory.id, context_zones)
        evaluation = _evaluate_checker(
            checker,
            response,
            query=query,
            memory=memory,
            event_context=event_context,
        )
        conflicts = [
            conflict
            for conflict in _detect_conflicts(self._protected_items())
            if memory.id in conflict["memory_ids"]
        ]

        event = ComplianceEvent(
            id=str(uuid.uuid4()),
            memory_id=memory.id,
            checker=checker,
            query=query,
            response=response,
            passed=evaluation["status"] == "passed",
            score=evaluation["score"],
            source=source,
            details={
                "zone": zone,
                "expected_check": expected_check,
                "memory_loaded": zone is not None,
                "status": evaluation["status"],
                "confidence": evaluation.get("confidence"),
                "resolution": evaluation.get("resolution"),
                "violations": evaluation["violations"],
                "conflict_topics": [conflict["topic"] for conflict in conflicts],
                "event_context": event_context,
            },
            created_at=_utcnow(),
        )
        self.store.add_event(self.agent_id, event)

        return {
            "memory_id": memory.id,
            "instruction": memory.content,
            "query": query,
            "checker": checker,
            "candidate_response": response,
            "memory_loaded": zone is not None,
            "zone": zone,
            "status": evaluation["status"],
            "passed": evaluation["status"] == "passed",
            "score": evaluation["score"],
            "confidence": evaluation.get("confidence"),
            "resolution": evaluation.get("resolution"),
            "violations": evaluation["violations"],
            "conflicts": conflicts,
            "event_id": event.id,
            "source": source,
        }

    def _active_items(self) -> List[MemoryItem]:
        return [
            item
            for item in self.store.list(self.agent_id)
            if item.status in {MemoryStatus.ACTIVE, MemoryStatus.CONFLICTED, MemoryStatus.STALE}
        ]

    def get_protected(self) -> List[MemoryItem]:
        """Return all protected (CORE/IDENTITY) memories, sorted by persistence score."""
        return self._protected_items()

    def _protected_items(self) -> List[MemoryItem]:
        items = [item for item in self._active_items() if is_protected(item.priority)]
        return sorted(items, key=lambda item: (-item.persistence_score, item.created_at))

    def _buffer_items(self) -> List[MemoryItem]:
        items = [
            item
            for item in self._active_items()
            if item.priority in {MemoryPriority.EPISODE, MemoryPriority.TRIVIAL}
        ]
        return sorted(items, key=lambda item: item.created_at, reverse=True)

    def _calculate_reinforcement(self, content: str, metadata: Dict[str, Any]) -> float:
        normalized = normalize_text(content)
        topic = metadata.get("topic")
        score = 1.0
        for item in self.store.list(self.agent_id):
            if normalize_text(item.content) == normalized:
                score += 1.0
            elif topic and item.metadata.get("topic") == topic:
                score += 0.5
        return score

    def _calculate_persistence(self, priority: MemoryPriority, reinforcement_score: float) -> float:
        base = {
            MemoryPriority.CORE: 1.0,
            MemoryPriority.IDENTITY: 0.8,
            MemoryPriority.PROJECT: 0.6,
            MemoryPriority.EPISODE: 0.35,
            MemoryPriority.TRIVIAL: 0.15,
        }[priority]
        return round(base + min(reinforcement_score * 0.15, 0.75), 3)

    def _mark_used(self, items: List[MemoryItem]) -> None:
        now = _utcnow()
        for item in items:
            updated = replace(item, last_used_at=now, updated_at=now)
            self.store.update(self.agent_id, updated)


def _resolve_storage_path(agent_id: str, storage_path: Optional[str]) -> Path:
    if storage_path is None:
        return Path.cwd() / ".memguard" / f"{agent_id}.sqlite3"
    candidate = Path(storage_path).expanduser()
    if candidate.suffix in {".db", ".sqlite", ".sqlite3"}:
        return candidate
    return candidate / f"{agent_id}.sqlite3"


def _render_context(
    protected: List[MemoryItem],
    active: List[MemoryItem],
    buffer_items: List[MemoryItem],
) -> str:
    sections: List[str] = []
    if protected:
        sections.append(_render_zone("Protected Zone", protected))
    if active:
        sections.append(_render_zone("Active Zone", active))
    if buffer_items:
        sections.append(_render_zone("Buffer Zone", buffer_items))
    return "\n\n".join(sections)


def _render_zone(title: str, items: List[MemoryItem]) -> str:
    lines = [f"{title}:"]
    for item in items:
        lines.append(f"- [{item.priority.value}] {item.content}")
    return "\n".join(lines)


def _fit_items_to_budget(items: List[MemoryItem], budget: int) -> List[MemoryItem]:
    return _fit_items_to_budget_detailed(items, budget)["selected"]


def _fit_items_to_budget_detailed(items: List[MemoryItem], budget: int) -> Dict[str, Any]:
    selected: List[MemoryItem] = []
    omitted: List[Dict[str, Any]] = []
    used = 0
    for item in items:
        item_cost = _estimate_tokens(item.content)
        if selected and used + item_cost > budget:
            omitted.append(
                {
                    "item": item,
                    "estimated_tokens": item_cost,
                    "reason": "budget_exceeded",
                }
            )
            continue
        selected.append(item)
        used += item_cost
    return {
        "selected": selected,
        "omitted": omitted,
        "used_tokens": used,
        "candidate_tokens": sum(_estimate_tokens(item.content) for item in items),
    }


def _selection_overflow_summary(*, zone: str, fit: Dict[str, Any], budget: int) -> Dict[str, Any]:
    used_tokens = int(fit["used_tokens"])
    omitted = list(fit["omitted"])
    return {
        "zone": zone,
        "budget": budget,
        "used_tokens": used_tokens,
        "candidate_tokens": int(fit["candidate_tokens"]),
        "selected_count": len(fit["selected"]),
        "omitted_count": len(omitted),
        "overflowed": bool(omitted) or used_tokens > budget,
        "omitted": [
            {
                "id": entry["item"].id,
                "instruction": entry["item"].content,
                "estimated_tokens": entry["estimated_tokens"],
                "reason": f"{zone}_budget_exceeded",
            }
            for entry in omitted
        ],
    }


def _estimate_tokens(text: str) -> int:
    return max(len(text) // 4, 1)


def _detect_conflicts(items: List[MemoryItem]) -> List[Dict[str, Any]]:
    by_topic: Dict[str, List[MemoryItem]] = {}
    for item in items:
        topic = item.metadata.get("topic")
        if topic is None:
            continue
        key = topic.split(":")[0]
        by_topic.setdefault(key, []).append(item)

    conflicts: List[Dict[str, Any]] = []
    for topic, topic_items in by_topic.items():
        unique_topics = {item.metadata.get("topic") for item in topic_items}
        if len(unique_topics) <= 1:
            continue
        latest = sorted(topic_items, key=lambda item: item.updated_at)[-1]
        conflicts.append(
            {
                "topic": topic,
                "memory_ids": [item.id for item in topic_items],
                "messages": [item.content for item in topic_items],
                "current_effective": latest.content,
            }
        )
    return conflicts


def _locate_memory_zone(memory_id: str, zones: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
    for zone_name, entries in zones.items():
        if any(entry["id"] == memory_id for entry in entries):
            return zone_name
    return None


def _collect_loaded_memories(
    agent_id: str,
    store: SQLiteStore,
    zones: Dict[str, List[Dict[str, Any]]],
    include_active: bool,
) -> Dict[str, MemoryItem]:
    candidate_zone_names = ["protected"]
    if include_active:
        candidate_zone_names.append("active")

    memories: Dict[str, MemoryItem] = {}
    for zone_name in candidate_zone_names:
        for entry in zones.get(zone_name, []):
            memory = store.get(agent_id, entry["id"])
            if memory is not None:
                memories[memory.id] = memory
    return memories


def _resolve_checker(memory: MemoryItem, expected_check: Optional[str]) -> str:
    if expected_check:
        return expected_check

    explicit_checker = memory.metadata.get("checker")
    if explicit_checker:
        return explicit_checker

    topic = memory.metadata.get("topic")
    if topic == "speech_level:informal":
        return "korean_informal"
    if topic == "speech_level:formal":
        return "korean_formal"
    if topic == "approval_before_action:required":
        return "approval_before_action"
    if topic == "code_block_language_tag:required":
        return "codeblock_language_tag"
    if topic == "git_destructive:forbidden":
        return "no_destructive_git_commands"

    if memory.kind == MemoryKind.GUARDRAIL:
        return "no_destructive_git_commands"
    if memory.kind == MemoryKind.INSTRUCTION:
        return "memory_loaded"
    return "memory_loaded"


def _checker_applicability(
    memory: MemoryItem,
    checker: str,
    query: str,
    response: str,
    *,
    event_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    custom = _CUSTOM_CHECKERS.get(checker)
    if custom is not None:
        applies_to = custom.get("applies_to")
        if applies_to is None:
            return {"applicable": True, "reason": "custom_checker_default"}
        result = applies_to(query=query, response=response, memory=memory, event_context=event_context)
        if isinstance(result, dict):
            return {
                "applicable": bool(result.get("applicable")),
                "reason": str(result.get("reason", "custom_checker")),
            }
        return {
            "applicable": bool(result),
            "reason": "custom_checker"
            if result
            else "custom_checker_not_applicable",
        }

    lowered_query = query.casefold()
    lowered_response = response.casefold()
    topic = memory.metadata.get("topic")
    event_type = (event_context or {}).get("event_type", "response")

    if checker in {"korean_informal", "korean_formal"}:
        if event_type == "action":
            return {"applicable": False, "reason": "response_text_required"}
        if _contains_hangul(response):
            return {"applicable": True, "reason": "korean_response_detected"}
        return {"applicable": False, "reason": "response_not_korean"}

    if checker == "codeblock_language_tag":
        if event_type == "action":
            return {"applicable": False, "reason": "response_text_required"}
        code_terms = ("코드", "예제", "snippet", "code", "python", "javascript", "typescript", "bash", "sql")
        if "```" in response or any(term in lowered_query for term in code_terms):
            return {"applicable": True, "reason": "code_response_expected"}
        return {"applicable": False, "reason": "no_code_signal"}

    if checker == "approval_before_action":
        if event_type != "action":
            return {"applicable": False, "reason": "action_event_required"}
        if _action_requires_approval(query, event_context):
            return {"applicable": True, "reason": "approval_required_action"}
        return {"applicable": False, "reason": "no_approval_required_action"}

    if checker == "no_destructive_git_commands":
        if event_type == "action":
            command = str((event_context or {}).get("command") or "").casefold()
            action_text = str((event_context or {}).get("action", "")).casefold()
            if command or "git" in action_text or "git" in lowered_query:
                return {"applicable": True, "reason": "git_action_detected"}
            return {"applicable": False, "reason": "not_git_related"}
        git_terms = ("git", "repo", "repository", "브랜치", "커밋", "checkout", "reset", "파일 수정")
        if any(term in lowered_query for term in git_terms) or any(term in lowered_response for term in git_terms):
            return {"applicable": True, "reason": "git_related_request"}
        return {"applicable": False, "reason": "not_git_related"}

    if checker == "memory_loaded":
        if topic:
            return {"applicable": False, "reason": "no_rule_based_checker_for_topic"}
        return {"applicable": False, "reason": "no_rule_based_checker"}

    return {"applicable": False, "reason": "unknown_checker"}


def _simulate_response(memory: MemoryItem, checker: str, test_input: str) -> str:
    if checker == "korean_informal":
        return "좋아, 바로 해볼게."
    if checker == "korean_formal":
        return "좋습니다. 바로 진행하겠습니다."
    if checker == "codeblock_language_tag":
        return "```python\nprint('hello')\n```"
    if checker == "no_destructive_git_commands":
        return "먼저 git status로 변경 사항을 확인해."
    return "Memory loaded for: {instruction}\nUser request: {query}".format(
        instruction=memory.content,
        query=test_input,
    )


def _evaluate_checker(
    checker: str,
    response: str,
    *,
    query: str = "",
    memory: Optional[MemoryItem] = None,
    event_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    custom = _CUSTOM_CHECKERS.get(checker)
    if custom is not None:
        result = custom["evaluate"](query=query, response=response, memory=memory, event_context=event_context)
        status = str(result.get("status", "passed" if result.get("passed") else "failed")).strip().casefold()
        if status not in {"passed", "failed", "uncertain"}:
            status = "passed" if result.get("passed") else "failed"
        raw_violations = result.get("violations", [])
        if isinstance(raw_violations, str):
            violations = [raw_violations]
        else:
            violations = list(raw_violations)
        return {
            "status": status,
            "passed": status == "passed",
            "score": float(result.get("score", 1.0 if result.get("passed") else 0.0)),
            "confidence": float(result.get("confidence", result.get("score", 1.0 if result.get("passed") else 0.0))),
            "violations": violations,
            "resolution": str(result.get("resolution", "custom_checker")),
        }

    violations: List[str] = []
    score = 1.0
    lowered = response.casefold()

    if checker == "korean_informal":
        if "습니다" in response or "어요" in response or "드립니다" in response:
            violations.append("formal_korean_detected")
            score = 0.0
    elif checker == "korean_formal":
        if not any(token in response for token in ("요", "습니다", "드립니다")):
            violations.append("formal_marker_missing")
            score = 0.0
    elif checker == "codeblock_language_tag":
        if not re.search(r"```[a-zA-Z0-9_+-]+\n", response):
            violations.append("language_tag_missing")
            score = 0.0
    elif checker == "approval_before_action":
        action_context = event_context or {}
        requires_approval = _action_requires_approval(query, action_context)
        approval_requested = bool(action_context.get("approval_requested"))
        approval_granted = bool(action_context.get("approval_granted"))
        executed = bool(action_context.get("executed", True))
        if requires_approval and not approval_requested:
            violations.append("approval_request_missing")
        if requires_approval and executed and not approval_granted:
            violations.append("executed_without_approval")
        if violations:
            score = 0.0
    elif checker == "no_destructive_git_commands":
        dangerous_patterns = (
            "git reset --hard",
            "git clean -fd",
            "git clean -xdf",
            "git checkout --",
            "rm -rf",
        )
        action_context = event_context or {}
        command_text = str(action_context.get("command", "")).casefold()
        matched = [pattern for pattern in dangerous_patterns if pattern in lowered or pattern in command_text]
        if matched:
            violations.extend(["destructive_command:" + pattern for pattern in matched])
            score = 0.0

    return {
        "status": "passed" if score >= 1.0 else "failed",
        "passed": score >= 1.0,
        "score": score,
        "confidence": 1.0,
        "violations": violations,
        "resolution": "rule_based",
    }


def _contains_hangul(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text))


def _action_requires_approval(query: str, event_context: Optional[Dict[str, Any]]) -> bool:
    context = event_context or {}
    if bool(context.get("requires_approval")):
        return True
    action_text = " ".join(
        str(context.get(key, "")) for key in ("action", "target", "command")
    ).casefold()
    query_text = query.casefold()
    approval_terms = ("수정", "edit", "rewrite", "overwrite", "변경", "write_file", "edit_file")
    return any(term in query_text or term in action_text for term in approval_terms)


def _render_action_event(event_context: Dict[str, Any]) -> str:
    parts = [
        "action={value}".format(value=event_context.get("action", "")),
        "target={value}".format(value=event_context.get("target", "")),
        "command={value}".format(value=event_context.get("command", "")),
        "requires_approval={value}".format(value=bool(event_context.get("requires_approval"))),
        "approval_requested={value}".format(value=bool(event_context.get("approval_requested"))),
        "approval_granted={value}".format(value=bool(event_context.get("approval_granted"))),
        "executed={value}".format(value=bool(event_context.get("executed", True))),
    ]
    return "ACTION " + " ".join(parts)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
