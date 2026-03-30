# Compliance Event Export & Observability

MemGuard stores every compliance check as a `ComplianceEvent` in local SQLite. This guide shows how to export, inspect, and feed those events into external tools.

## What Gets Stored

Every call to `check()`, `observe_action()`, or `replay()` creates a compliance event:

```python
ComplianceEvent(
    id="uuid",
    memory_id="uuid",          # which instruction was checked
    checker="korean_informal",  # which checker ran
    query="오늘 뭐 해?",        # user input
    response="좋아, 하자.",      # agent output
    passed=True,
    score=1.0,
    source="guard_check",       # check / guard_action / replay / observe
    details={                   # zone, violations, conflicts, etc.
        "zone": "protected",
        "violations": [],
        "conflict_topics": [],
    },
    created_at="2026-03-30T...",
)
```

## Export Memories

```python
from memguard import MemoryGuard

guard = MemoryGuard(agent_id="my-agent", storage_path="./data")

# Export all memories to JSON
guard.export("./exports/memories.json")
```

## Export Compliance Events

Compliance events are accessible through the low-level `Memory` API:

```python
from memguard import MemoryGuard

guard = MemoryGuard(agent_id="my-agent", storage_path="./data")

# All events
events = guard._mem.store.list_events("my-agent")

# Events for a specific instruction
events = guard._mem.store.list_events("my-agent", memory_id="<uuid>")

# Convert to dicts for JSON export
import json
from pathlib import Path

payload = [event.to_dict() for event in events]
Path("./exports/compliance_events.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2)
)
```

## JSONL Streaming for Log Pipelines

For production logging, write one event per line:

```python
import json

events = guard._mem.store.list_events("my-agent")
with open("./logs/compliance.jsonl", "a") as f:
    for event in events:
        f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
```

This format is compatible with:
- **Datadog Log Management** — ingest JSONL directly
- **Elasticsearch / OpenSearch** — Filebeat reads JSONL
- **Grafana Loki** — promtail can parse JSON lines
- **AWS CloudWatch** — structured JSON log entries

## Quick Inspection Script

```python
"""Quick compliance dashboard — run after a session."""
from memguard import MemoryGuard

guard = MemoryGuard(agent_id="my-agent", storage_path="./data")
report = guard.report()

print(f"Protected instructions: {report['protected']}")
print(f"Compliance rate:        {report['compliance_rate']}")
print(f"Drift warnings:        {report['drift_warnings']}")
print(f"Conflicts:             {report['conflicts']}")

# List recent failures
events = guard._mem.store.list_events("my-agent")
failures = [e for e in events if not e.passed]
for f in failures[-5:]:
    print(f"\n❌ {f.checker} failed on memory {f.memory_id[:8]}...")
    print(f"   Query: {f.query[:60]}")
    print(f"   Score: {f.score}")
```

## SQLite Direct Access

The compliance events table can also be queried directly for custom analytics:

```sql
-- Compliance rate over time (daily)
SELECT
    date(created_at) as day,
    COUNT(*) as total,
    SUM(passed) as passed,
    ROUND(CAST(SUM(passed) AS FLOAT) / COUNT(*), 3) as rate
FROM compliance_events
WHERE agent_id = 'my-agent'
GROUP BY date(created_at)
ORDER BY day;

-- Most frequently failing instructions
SELECT
    memory_id,
    checker,
    COUNT(*) as checks,
    SUM(CASE WHEN passed = 0 THEN 1 ELSE 0 END) as failures
FROM compliance_events
WHERE agent_id = 'my-agent'
GROUP BY memory_id, checker
HAVING failures > 0
ORDER BY failures DESC;
```

The database file lives at the `storage_path` you configured (default: `.memguard/<agent_id>.sqlite3`).
