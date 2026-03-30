# 3-Zone Token Budget Policy

InstructionGuard allocates context budget across three zones, each with a different eviction policy.

## Allocation Ratios

When `build_context(query, token_budget=N)` is called, the budget splits into:

| Zone | Share | Eviction Policy |
|------|-------|-----------------|
| **Protected** | 50% | Never evicted. Ordered by `persistence_score` (descending), then `created_at`. |
| **Active** | 35% | Relevance-ranked via `recall()`. Only non-protected items. |
| **Buffer** | 15% (remainder) | Most recent episodes and trivial items. Newest first. |

Example with `token_budget=4000`:

```
Protected: 2000 tokens  (core + identity memories)
Active:    1400 tokens  (query-relevant project/fact memories)
Buffer:     600 tokens  (recent episodes, compressible)
```

## Token Estimation

The current implementation uses a lightweight heuristic:

```python
tokens ≈ max(len(text) // 4, 1)
```

This is a rough character-to-token ratio that works reasonably well for mixed Korean/English text. It is **not** provider-specific tokenization (no tiktoken, no SentencePiece). The heuristic keeps the library zero-dependency but may over- or under-count by ~20% compared to actual model tokenizers.

## How Items Are Selected

Each zone runs `_fit_items_to_budget()`:

1. Items are pre-sorted (by persistence score, relevance score, or recency depending on zone).
2. Items are added one by one until the zone budget is exhausted.
3. Items that would exceed the remaining budget are **skipped** (not truncated).

This means the first item always gets in (even if it alone exceeds the budget), but subsequent items must fit.

## What Happens When Protected Zone Overflows

**Current behavior (v0.3):** If protected instructions exceed 50% of the total budget, lower-`persistence_score` items may be omitted from the final prompt. InstructionGuard now reports that overflow explicitly, but there is still no compact rendering or semantic summarization step.

**Known limitation:** With many protected instructions (20+), some will still be omitted under pressure. See [#9](https://github.com/quanttraderkim/instructionguard/issues/9) for compact rendering under budget pressure and [#13](https://github.com/quanttraderkim/instructionguard/issues/13) for configurable zone budget policy.

## Persistence Score

Each memory gets a persistence score that determines its priority within the protected zone:

```
persistence = base_score + min(reinforcement × 0.15, 0.75)
```

Base scores by priority level:

| Priority | Base Score |
|----------|-----------|
| `core` | 1.0 |
| `identity` | 0.8 |
| `project` | 0.6 |
| `episode` | 0.35 |
| `trivial` | 0.15 |

Reinforcement increases when the same content or topic is stored multiple times.

## Future Evolution

- **Compact rendering** ([#9](https://github.com/quanttraderkim/instructionguard/issues/9)): Summarize or bullet-compress protected instructions when budget is tight.
- **Configurable zone policy** ([#13](https://github.com/quanttraderkim/instructionguard/issues/13)): Let users tune the protected / active / buffer split.
- **Saturation benchmark** ([#11](https://github.com/quanttraderkim/instructionguard/issues/11)): Test ISR with 20-50 protected instructions to quantify real-world limits.
- **Provider-specific tokenizers**: Optional tiktoken/SentencePiece integration for accurate counts.
