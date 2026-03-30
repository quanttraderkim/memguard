from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = Path(__file__).with_name("latest_results.json")
OUTPUT_PATH = ROOT / "assets" / "isb-summary.svg"
TARGET_BUDGET = 500

STRATEGY_ORDER = ["no_memory", "naive_fifo", "pinned_prompt", "instructionguard"]
COLORS = {
    "no_memory": "#64748B",
    "naive_fifo": "#94A3B8",
    "pinned_prompt": "#E7A83B",
    "instructionguard": "#48C78E",
}
LABELS = {
    "no_memory": "No Memory",
    "naive_fifo": "Naive FIFO",
    "pinned_prompt": "Pinned Prompt",
    "instructionguard": "InstructionGuard",
}


def load_budget_rows() -> dict[str, dict]:
    payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    rows = {}
    for strategy, entries in payload["persistence"].items():
        for entry in entries:
            if entry["token_budget"] == TARGET_BUDGET:
                rows[strategy] = entry
                break
    return rows


def render_svg(rows: dict[str, dict]) -> str:
    width = 1200
    height = 540
    left = 80
    base_y = 420
    bar_width = 180
    gap = 32
    max_bar_height = 180

    def bar_height(value: float) -> float:
        return max_bar_height * value

    parts = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" fill="none" xmlns="http://www.w3.org/2000/svg">',
        '<rect width="1200" height="540" rx="28" fill="#0F1724"/>',
        '<rect x="36" y="34" width="1128" height="472" rx="22" fill="#121E2D" stroke="#25384C"/>',
        '<text x="80" y="96" fill="#F8FAFC" font-size="34" font-weight="700" font-family="ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">Instruction survival at the hardest budget</text>',
        '<text x="80" y="132" fill="#9FB0C5" font-size="18" font-weight="500" font-family="ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">Bundled ISB snapshot at token budget 500. InstructionGuard keeps both critical instructions and active project facts alive.</text>',
        f'<line x1="{left}" y1="{base_y}" x2="1100" y2="{base_y}" stroke="#36506A" stroke-width="2"/>',
        f'<line x1="{left}" y1="190" x2="{left}" y2="{base_y}" stroke="#36506A" stroke-width="2"/>',
    ]

    for index, strategy in enumerate(STRATEGY_ORDER):
        row = rows[strategy]
        x = left + 36 + index * (bar_width + gap)
        tsr = float(row["task_success_rate"])
        afr = float(row["active_fact_retention"])
        isr = float(row["instruction_survival_rate"])
        h = bar_height(tsr)
        y = base_y - h
        parts.extend(
            [
                f'<rect x="{x}" y="{y}" width="{bar_width}" height="{h}" rx="18" fill="{COLORS[strategy]}"/>',
                f'<text x="{x + bar_width / 2}" y="454" fill="#E8EEF5" font-size="19" font-weight="700" text-anchor="middle" font-family="ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">{LABELS[strategy]}</text>',
                f'<text x="{x + bar_width / 2}" y="{y - 14}" fill="#F8FAFC" font-size="20" font-weight="700" text-anchor="middle" font-family="ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">TSR {tsr:.3g}</text>',
                f'<text x="{x + bar_width / 2}" y="482" fill="#AFC0D3" font-size="15" font-weight="600" text-anchor="middle" font-family="ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">ISR {isr:.3g} · AFR {afr:.3g}</text>',
            ]
        )

    parts.extend(
        [
            '<rect x="820" y="176" width="294" height="112" rx="18" fill="#162637" stroke="#2D475F"/>',
            '<text x="848" y="214" fill="#F8FAFC" font-size="19" font-weight="700" font-family="ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">Why this matters</text>',
            '<text x="848" y="246" fill="#B7C6D6" font-size="15" font-weight="500" font-family="ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">Pinned prompts keep rules alive, but they lose</text>',
            '<text x="848" y="268" fill="#B7C6D6" font-size="15" font-weight="500" font-family="ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">active project facts. InstructionGuard preserves both.</text>',
        ]
    )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    rows = load_budget_rows()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render_svg(rows), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
