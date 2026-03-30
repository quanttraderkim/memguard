from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .models import ComplianceEvent, MemoryItem, MemoryKind, MemoryPriority, MemoryStatus


class SQLiteStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                content TEXT NOT NULL,
                priority TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                source TEXT NOT NULL,
                scope TEXT NOT NULL,
                reinforcement_score REAL NOT NULL,
                persistence_score REAL NOT NULL,
                last_used_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memories_agent_id ON memories(agent_id);
            CREATE INDEX IF NOT EXISTS idx_memories_priority ON memories(agent_id, priority);
            CREATE TABLE IF NOT EXISTS compliance_events (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                checker TEXT NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                passed INTEGER NOT NULL,
                score REAL NOT NULL,
                source TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_compliance_agent_id ON compliance_events(agent_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_compliance_memory_id ON compliance_events(agent_id, memory_id, created_at);
            """
        )
        self.conn.commit()

    def add(self, agent_id: str, item: MemoryItem) -> None:
        self.conn.execute(
            """
            INSERT INTO memories (
                id, agent_id, content, priority, kind, status, source, scope,
                reinforcement_score, persistence_score, last_used_at,
                created_at, updated_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.id,
                agent_id,
                item.content,
                item.priority.value,
                item.kind.value,
                item.status.value,
                item.source,
                item.scope,
                item.reinforcement_score,
                item.persistence_score,
                item.last_used_at.isoformat() if item.last_used_at else None,
                item.created_at.isoformat(),
                item.updated_at.isoformat(),
                json.dumps(item.metadata, ensure_ascii=False, sort_keys=True),
            ),
        )
        self.conn.commit()

    def list(self, agent_id: str) -> List[MemoryItem]:
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE agent_id = ? ORDER BY created_at ASC", (agent_id,)
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def get(self, agent_id: str, memory_id: str) -> Optional[MemoryItem]:
        row = self.conn.execute(
            "SELECT * FROM memories WHERE agent_id = ? AND id = ?", (agent_id, memory_id)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_item(row)

    def delete(self, agent_id: str, memory_id: str) -> None:
        self.conn.execute("DELETE FROM memories WHERE agent_id = ? AND id = ?", (agent_id, memory_id))
        self.conn.commit()

    def update(self, agent_id: str, item: MemoryItem) -> None:
        self.conn.execute(
            """
            UPDATE memories
            SET content = ?, priority = ?, kind = ?, status = ?, source = ?, scope = ?,
                reinforcement_score = ?, persistence_score = ?, last_used_at = ?,
                created_at = ?, updated_at = ?, metadata_json = ?
            WHERE agent_id = ? AND id = ?
            """,
            (
                item.content,
                item.priority.value,
                item.kind.value,
                item.status.value,
                item.source,
                item.scope,
                item.reinforcement_score,
                item.persistence_score,
                item.last_used_at.isoformat() if item.last_used_at else None,
                item.created_at.isoformat(),
                item.updated_at.isoformat(),
                json.dumps(item.metadata, ensure_ascii=False, sort_keys=True),
                agent_id,
                item.id,
            ),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def add_event(self, agent_id: str, event: ComplianceEvent) -> None:
        self.conn.execute(
            """
            INSERT INTO compliance_events (
                id, agent_id, memory_id, checker, query, response, passed,
                score, source, details_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                agent_id,
                event.memory_id,
                event.checker,
                event.query,
                event.response,
                int(event.passed),
                event.score,
                event.source,
                json.dumps(event.details, ensure_ascii=False, sort_keys=True),
                event.created_at.isoformat(),
            ),
        )
        self.conn.commit()

    def list_events(self, agent_id: str, memory_id: Optional[str] = None) -> List[ComplianceEvent]:
        if memory_id is None:
            rows = self.conn.execute(
                "SELECT * FROM compliance_events WHERE agent_id = ? ORDER BY created_at ASC",
                (agent_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT * FROM compliance_events
                WHERE agent_id = ? AND memory_id = ?
                ORDER BY created_at ASC
                """,
                (agent_id, memory_id),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def _row_to_item(self, row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            content=row["content"],
            priority=MemoryPriority(row["priority"]),
            kind=MemoryKind(row["kind"]),
            status=MemoryStatus(row["status"]),
            source=row["source"],
            scope=row["scope"],
            reinforcement_score=float(row["reinforcement_score"]),
            persistence_score=float(row["persistence_score"]),
            last_used_at=datetime.fromisoformat(row["last_used_at"]) if row["last_used_at"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            metadata=json.loads(row["metadata_json"]),
        )

    def _row_to_event(self, row: sqlite3.Row) -> ComplianceEvent:
        return ComplianceEvent(
            id=row["id"],
            memory_id=row["memory_id"],
            checker=row["checker"],
            query=row["query"],
            response=row["response"],
            passed=bool(row["passed"]),
            score=float(row["score"]),
            source=row["source"],
            details=json.loads(row["details_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )
