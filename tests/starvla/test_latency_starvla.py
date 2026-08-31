from __future__ import annotations

import numpy as np

from eval.simpler_env.runners.latency_starvla import summarize, unnormalize


def test_summarize_interpolates_percentiles() -> None:
    result = summarize([1.0, 2.0, 3.0, 4.0])

    assert result == {
        "count": 4,
        "avg": 2.5,
        "min": 1.0,
        "p50": 2.5,
        "p90": 3.7,
        "p99": 3.97,
        "max": 4.0,
    }


def test_unnormalize_uses_checkpoint_default_profile() -> None:
    normalized = np.zeros((1, 16, 7), dtype=np.float32)
    normalized[..., 6] = 0.75
    profile = {
        "action": {
            "q01": [-2.0] * 6 + [0.0],
            "q99": [4.0] * 6 + [1.0],
            "mask": [True] * 6 + [False],
        }
    }

    actual = unnormalize(normalized, {"bridge": profile, "unused": profile})

    np.testing.assert_array_equal(actual[..., :6], 1.0)
    np.testing.assert_array_equal(actual[..., 6], 1.0)
