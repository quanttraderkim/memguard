from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class StrEnum(str, Enum):
    pass


class MemoryPriority(StrEnum):
    CORE = "core"
    IDENTITY = "identity"
    PROJECT = "project"
    EPISODE = "episode"
    TRIVIAL = "trivial"


class MemoryKind(StrEnum):
    INSTRUCTION = "instruction"
    GUARDRAIL = "guardrail"
    PREFERENCE = "preference"
    FACT = "fact"
    SUMMARY = "summary"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    STALE = "stale"
    CONFLICTED = "conflicted"
    ARCHIVED = "archived"


@dataclass
class MemoryItem:
    id: str
    content: str
    priority: MemoryPriority
    kind: MemoryKind
    status: MemoryStatus
    source: str
    scope: str
    reinforcement_score: float
    persistence_score: float
    last_used_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["priority"] = self.priority.value
        payload["kind"] = self.kind.value
        payload["status"] = self.status.value
        payload["created_at"] = self.created_at.isoformat()
        payload["updated_at"] = self.updated_at.isoformat()
        payload["last_used_at"] = self.last_used_at.isoformat() if self.last_used_at else None
        return payload


@dataclass
class ComplianceEvent:
    id: str
    memory_id: str
    checker: str
    query: str
    response: str
    passed: bool
    score: float
    source: str
    details: Dict[str, Any]
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return payload
