# MemGuard

Local-first instruction persistence and memory integrity for AI agents.

> Keep critical instructions alive, observable, and testable across compaction, tool calls, and drift.

## The Problem

Every AI agent has memory now. But storing a memory and **actually following it** are two different things.

```
User: "항상 반말로 대답해"
Agent: ✅ Saved to memory

... 50 turns later, context compaction happens ...

User: "오늘 뭐 할까?"
Agent: "좋습니다. 일정을 확인해보겠습니다."  ← 🤦 forgot the instruction
```

This happens because:
- Context windows compress old content
- Memory retrieval misses critical instructions
- No one verifies if stored memories actually influence behavior

## How It Works

MemGuard protects instructions across three mechanisms:

1. **Memory Guardian** — Critical instructions get a protected zone in context that survives compaction
2. **Compliance Check** — Verify AI responses and execution events actually follow stored instructions
3. **Drift Detection** — Detect when instruction compliance degrades over time

## Quickstart

```bash
python -m pip install -e ".[dev]"
```

```python
from memguard import MemoryGuard

guard = MemoryGuard(agent_id="my-agent")

# Protect critical instructions
guard.protect("항상 반말로 대답해")
guard.protect("코드 블록에 언어 태그 필수")
guard.protect("git reset --hard 절대 금지", kind="guardrail")

# Check AI responses
result = guard.check(
    query="오늘 뭐 할까?",
    response="좋습니다. 일정을 확인해보겠습니다.",
)
print(result["passed"])      # False
print(result["violations"])  # ["formal_korean_detected"]

# Generate reminder prompt for LLM
reminder = guard.reminder()
# "⚠️ 다음 지시를 반드시 준수하세요:\n- 항상 반말로 대답해\n- ..."

# Get integrity report
report = guard.report()
print(report["drift_warnings"])  # 0
print(report["observed_checks"]) # 1
print(report["compliance_rate"]) # 0.0
```

## Custom Checkers

You can extend MemGuard without waiting for a built-in rule:

```python
from memguard import MemoryGuard

guard = MemoryGuard(agent_id="my-agent")

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
print(result["status"])      # failed
print(result["violations"])  # ["korean_missing"]
```

Custom checkers receive `query`, `response`, `memory`, and `event_context`, so the same API can validate both text outputs and real execution events.

## Optional LLM Checkers

For open-ended instructions that do not map well to a rule, you can register an LLM-backed checker.
`provider` can be `openai`, `anthropic`, or `gemini`, and if you omit it MemGuard will infer it from the model name:

```python
import os

from memguard import MemoryGuard

guard = MemoryGuard(
    agent_id="my-agent",
    llm="claude-3-5-sonnet-latest",
    llm_api_key=os.getenv("ANTHROPIC_API_KEY"),
)

guard.register_llm_checker(
    "semantic_summary_rule",
    provider="anthropic",
    rubric_template="summary_first",
    rubric="Judge whether the assistant writes a one-line summary before the main answer.",
    negative_recheck=True,
)

guard.protect("답변 전에 한 줄 요약을 먼저 써", checker="semantic_summary_rule")
result = guard.check(
    query="이 PR 요약해줘",
    response="요약: 인증 버그를 수정한다.\n자세한 내용은...",
)
print(result["status"])  # passed
```

If you do not want a network call, pass a local `judge=` callable instead. Tests use that path so the core suite stays deterministic. For Gemini, pass `provider="gemini"` and a `gemini-*` model with `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

MemGuard now supports three stability levers for LLM checkers:
- `rubric_template`: use a narrower built-in judging template such as `language_compliance`, `summary_first`, `brevity_limit`, or `approval_before_action`
- `local_fallback=True`: compare the LLM decision against a deterministic local signal when one exists
- `uncertainty_threshold`: downgrade low-confidence negatives into `uncertain` instead of hard-failing immediately
- `negative_recheck=True`: run one extra confirmatory pass on LLM negatives before keeping a hard failure

LLM checkers are still experimental, but they now expose `passed`, `failed`, and `uncertain` instead of collapsing every ambiguous case into `failed`. That makes them much safer to use as an advisory layer before production rollout.

## Action-Level Verification

MemGuard can also verify what the agent actually did, not just what it said:

```python
from memguard import MemoryGuard

guard = MemoryGuard(agent_id="my-agent")
guard.protect("파일 수정 전에는 항상 먼저 허가를 요청해")
guard.protect("git reset --hard 절대 금지", kind="guardrail")

result = guard.observe_action(
    query="README 수정해줘",
    action="edit_file",
    target="README.md",
    requires_approval=True,
    approval_requested=False,
    approval_granted=False,
    executed=True,
)

print(result["status"])      # failed
print(result["violations"])  # ["approval_request_missing", "executed_without_approval"]
```

This makes MemGuard useful for tool-using coding agents where correctness depends on actual execution order, not just generated text.

`check()` now separates five outcomes:
- `passed`: an applicable checker ran and passed
- `failed`: an applicable checker ran and found a violation
- `uncertain`: an LLM-backed checker ran but the result stayed ambiguous or conflicted with local evidence
- `unverified`: no checker exists yet for that instruction
- `not_applicable`: a checker exists, but this turn did not require it

## Benchmark

MemGuard now ships with a bundled `Instruction Survival Benchmark (ISB)`:

```bash
python benchmarks/isb/run.py --output benchmarks/isb/latest_results.json
```

Optional provider-backed open-ended track:

```bash
python benchmarks/isb/run.py \
  --llm-provider anthropic \
  --llm-model claude-3-5-sonnet-latest \
  --api-key "$ANTHROPIC_API_KEY" \
  --output benchmarks/isb/latest_results.json
```

The benchmark is split into two tracks:
- `Persistence`: compares `no_memory`, `naive_fifo`, `pinned_prompt`, and `memguard`
- `Verification`: measures precision / recall / F1 / false-positive rate / mean turns to detection
- `LLM Verification` optionally evaluates open-ended semantic rules across language, summary-first, brevity, and approval-before-action families

On the bundled deterministic suite, the key result at token budget `500` is:

| Strategy | ISR | Active Fact Retention | Task Success Rate |
|----------|-----|------------------------|-------------------|
| `no_memory` | `0.0` | `0.0` | `0.0` |
| `naive_fifo` | `0.0` | `0.0` | `0.0` |
| `pinned_prompt` | `1.0` | `0.0` | `0.714` |
| `memguard` | `1.0` | `1.0` | `1.0` |

The bundled verification track currently reports `precision=1.0`, `recall=1.0`, `f1=1.0`, `fpr=0.0`, and `mttd=1.0` on supported built-in rules. Full raw output is written to `benchmarks/isb/latest_results.json`.

The optional provider-backed LLM track is intentionally reported separately and now also reports `uncertain_rate`. The bundled open-ended suite currently covers 21 cases across four rubric families, so it is much more useful for rubric tuning than the earlier tiny smoke set. Treat those results as tuning feedback, not headline marketing numbers.

The open-ended benchmark dataset now lives in `benchmarks/isb/llm_cases.json`, so you can grow or fork the semantic evaluation set without changing benchmark code.

## Development

Run tests from a fresh checkout:

```bash
python -m pytest tests -v
```

For editable installs:

```bash
python -m pip install -e ".[dev]"
```

## Works With

- **Standalone** — no dependencies, local SQLite
- **Any agent framework** — LangChain, CrewAI, AutoGen, OpenClaw
- **Any memory backend** — use alongside mem0, Zep, or your own

## API

### MemoryGuard (high-level)

| Method | Description |
|--------|-------------|
| `protect(instruction)` | Register a protected instruction |
| `check(query, response)` | Verify response compliance |
| `observe_action(...)` | Verify execution-time compliance |
| `reminder()` | Generate reminder prompt for LLM |
| `report()` | Full integrity report |
| `context(query)` | Build context with protected zone |
| `remember/recall/forget` | Delegated to Memory |
| `verify/detect_drift` | Delegated to Memory |
| `export/stats` | Delegated to Memory |

### Memory (low-level)

Full control over memory storage, retrieval, replay, and consolidation.
See `src/memguard/core.py` for complete API.

## Architecture

```
┌─────────────────────────────────────┐
│           Your AI Agent             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│          MemoryGuard API            │
│  protect / check / reminder / report│
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│          Memory Core                │
│                                     │
│  ┌──────────┐ ┌──────────────────┐  │
│  │ 3-Zone   │ │ Compliance       │  │
│  │ Context  │ │ Engine           │  │
│  │ Builder  │ │ (replay/drift)   │  │
│  └──────────┘ └──────────────────┘  │
│                                     │
│  ┌──────────────────────────────┐   │
│  │ SQLite Store (local-first)   │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
```

### 3-Zone Context Protection

```
Context Window
├── 🔒 Protected Zone (never removed)
│   └── User instructions, guardrails, identity
├── 📋 Active Zone (relevance-based)
│   └── Current task context
└── 💭 Buffer Zone (compressible)
    └── Old conversations, trivial facts
```

## Status

v0.3 alpha — Local-first, rule-based response and action-level compliance checkers, optional experimental LLM-backed semantic checkers, zero mandatory external dependencies.

## License

Apache 2.0
