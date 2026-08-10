from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools" / "hf2gguf" / "starvla"
sys.path.insert(0, str(TOOLS_DIR))

from compare_starvla_actions import (  # noqa: E402
    ComparisonError,
    compare_actions,
    load_actions,
)


def test_action_gate_accepts_three_percent() -> None:
    reference = np.ones((16, 7), dtype=np.float64)
    result = compare_actions(reference, reference * 1.03)
    assert result == {
        "shape": [16, 7],
        "relative_l2": pytest.approx(0.03),
        "limit": 0.03,
        "passed": True,
    }


def test_action_gate_rejects_shape_nonfinite_and_excess_error(tmp_path: Path) -> None:
    reference = np.ones((16, 7), dtype=np.float64)
    with pytest.raises(ComparisonError, match="shape mismatch"):
        compare_actions(reference, np.ones((15, 7)))
    assert not compare_actions(reference, reference * 1.031)["passed"]

    path = tmp_path / "reference.json"
    path.write_text(
        json.dumps({"outputs": {"actions": [[float("nan")]]}}), encoding="utf-8"
    )
    with pytest.raises(ComparisonError, match="non-finite"):
        load_actions(path, "outputs.actions")


def test_loader_accepts_python_batch_dimension(tmp_path: Path) -> None:
    path = tmp_path / "reference.json"
    path.write_text(
        json.dumps({"outputs": {"actions": [[[1.0] * 7] * 16]}}), encoding="utf-8"
    )
    assert load_actions(path, "outputs.actions").shape == (16, 7)
