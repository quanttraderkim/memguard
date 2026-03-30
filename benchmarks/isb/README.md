# Instruction Survival Benchmark

This directory contains the bundled benchmark suite for MemGuard.

Run:

```bash
python benchmarks/isb/run.py --output benchmarks/isb/latest_results.json
```

Render the bundled SVG summary for the README:

```bash
python benchmarks/isb/render_svg.py
```

Optional open-ended LLM track:

```bash
python benchmarks/isb/run.py \
  --llm-provider anthropic \
  --llm-model claude-3-5-sonnet-latest \
  --api-key "$ANTHROPIC_API_KEY" \
  --output benchmarks/isb/latest_results.json
```

Tracks:

- `Persistence`: instruction survival, active fact retention, task success rate, protected token ratio
- `Verification`: precision, recall, F1, false-positive rate, mean turns to detection
- `LLM Verification`: open-ended semantic checks for language compliance, summary-first behavior, brevity limits, and approval-before-action, with `uncertain_rate` reported separately

The benchmark is deterministic unless you enable the optional LLM verification track.

The LLM track should be treated as experimental. Provider behavior, rubric wording, and JSON formatting stability can materially affect precision, false-positive rate, and uncertainty rate. The bundled open-ended suite is intentionally larger than the initial smoke test so you can tune rubric templates, negative recheck behavior, and hybrid fallback behavior instead of reading too much into a tiny sample.

The open-ended semantic cases live in `benchmarks/isb/llm_cases.json`.
The generated visual summary is written to `assets/isb-summary.svg`.
