from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_isb_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks" / "isb" / "run.py"
    spec = importlib.util.spec_from_file_location("instructionguard_isb_run", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_saturation_benchmark_exposes_overflow_at_50_and_500():
    module = _load_isb_module()
    results = module.run_saturation_benchmark()

    row = next(item for item in results["protected_50"]["instructionguard"] if item["token_budget"] == 500)

    assert row["overflowed"] is True
    assert row["protected_loaded"] < row["protected_candidates"]
    assert row["protected_omitted"] > 0


def test_saturation_benchmark_keeps_20_rules_at_1000_for_instructionguard():
    module = _load_isb_module()
    results = module.run_saturation_benchmark()

    row = next(item for item in results["protected_20"]["instructionguard"] if item["token_budget"] == 1000)

    assert row["overflowed"] is False
    assert row["protected_loaded"] == row["protected_candidates"]
    assert row["instruction_survival_rate"] == 1.0
