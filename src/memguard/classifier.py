from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .models import MemoryKind, MemoryPriority

INSTRUCTION_KEYWORDS = (
    "항상",
    "반드시",
    "하지 마",
    "하지마",
    "말고",
    "대답해",
    "붙여",
    "사용해",
    "써줘",
    "해줘",
    "do not",
    "don't",
    "always",
    "never",
    "use ",
    "reply ",
)

GUARDRAIL_KEYWORDS = (
    "금지",
    "하지 마",
    "삭제하지 마",
    "destructive",
    "do not",
    "never",
)

IDENTITY_KEYWORDS = ("내 이름", "이름은", "말투", "호칭", "nickname", "name is")
PROJECT_KEYWORDS = ("repo", "repository", "프로젝트", "저장소", "pnpm", "npm", "pytest", "vitest")
PREFERENCE_KEYWORDS = ("좋아", "선호", "prefer", "prefers", "취향")


def normalize_text(value: str) -> str:
    value = value.casefold().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def tokenize(value: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9가-힣_-]+", value.casefold())


def infer_memory_profile(
    content: str,
    *,
    priority: Optional[str],
    kind: Optional[str],
    metadata: Optional[Dict[str, Any]],
) -> Tuple[MemoryPriority, MemoryKind, Dict[str, Any]]:
    metadata = dict(metadata or {})
    lowered = normalize_text(content)

    inferred_kind = _infer_kind(lowered)
    inferred_priority = _infer_priority(lowered, metadata, inferred_kind)

    final_kind = MemoryKind(kind) if kind else inferred_kind
    final_priority = MemoryPriority(priority) if priority else inferred_priority

    topic = infer_topic(content)
    if topic:
        metadata.setdefault("topic", topic)

    return final_priority, final_kind, metadata


def infer_topic(content: str) -> Optional[str]:
    lowered = normalize_text(content)
    if "반말" in lowered:
        return "speech_level:informal"
    if "존댓말" in lowered:
        return "speech_level:formal"
    if any(term in lowered for term in ("허가", "승인", "permission", "approve")) and any(
        term in lowered for term in ("수정", "edit", "rewrite", "변경", "파일")
    ):
        return "approval_before_action:required"
    if ("언어 태그" in lowered or "language tag" in lowered) and any(
        term in lowered for term in ("항상", "반드시", "always")
    ):
        return "code_block_language_tag:required"
    if "destructive git" in lowered or ("git" in lowered and "하지 마" in lowered):
        return "git_destructive:forbidden"
    return None


def lexical_score(query: str, content: str, metadata: Dict[str, Any]) -> float:
    query_tokens = tokenize(query)
    content_tokens = tokenize(content)
    if not query_tokens or not content_tokens:
        return 0.0

    overlap = Counter(query_tokens) & Counter(content_tokens)
    overlap_score = sum(overlap.values()) / max(len(query_tokens), 1)

    bonus = 0.0
    repo = str(metadata.get("repo", "")).casefold()
    if repo and any(token in repo for token in query_tokens):
        bonus += 0.35

    return overlap_score + bonus


def is_protected(priority: MemoryPriority) -> bool:
    return priority in {MemoryPriority.CORE, MemoryPriority.IDENTITY}


def _infer_kind(content: str) -> MemoryKind:
    if any(keyword in content for keyword in GUARDRAIL_KEYWORDS):
        return MemoryKind.GUARDRAIL
    if any(keyword in content for keyword in INSTRUCTION_KEYWORDS):
        return MemoryKind.INSTRUCTION
    if any(keyword in content for keyword in PREFERENCE_KEYWORDS):
        return MemoryKind.PREFERENCE
    return MemoryKind.FACT


def _infer_priority(content: str, metadata: Dict[str, Any], kind: MemoryKind) -> MemoryPriority:
    if any(keyword in content for keyword in IDENTITY_KEYWORDS):
        return MemoryPriority.IDENTITY
    if metadata.get("repo") or any(keyword in content for keyword in PROJECT_KEYWORDS):
        return MemoryPriority.PROJECT
    if kind in {MemoryKind.INSTRUCTION, MemoryKind.GUARDRAIL}:
        return MemoryPriority.CORE
    if kind == MemoryKind.PREFERENCE:
        return MemoryPriority.IDENTITY
    return MemoryPriority.EPISODE
