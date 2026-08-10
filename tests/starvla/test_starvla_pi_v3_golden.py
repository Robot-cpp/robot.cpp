from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools" / "hf2gguf" / "starvla"
sys.path.insert(0, str(TOOLS_DIR))

from generate_starvla_pi_v3_golden import (  # noqa: E402
    CONDITIONING_TAP_NAMES,
    expected_model_instruction,
    expected_runtime_contract,
)


class PIv3ReferenceTest(unittest.TestCase):
    def test_instruction_template(self) -> None:
        config = {
            "datasets": {
                "vla_data": {"CoT_prompt": "Task: {instruction}"},
            }
        }
        self.assertEqual(expected_model_instruction(config, "grab block"), "Task: grab block")

    def test_action_oracle_contract(self) -> None:
        contract = expected_runtime_contract()
        self.assertEqual(contract["conditioning"]["hidden_tuple_indices"], list(range(1, 37)))
        self.assertEqual(contract["conditioning"]["hidden_tap_names"], CONDITIONING_TAP_NAMES)
        self.assertEqual(contract["timesteps"], [0, 250, 500, 750])
        self.assertEqual(contract["action_shape"], [16, 7])


if __name__ == "__main__":
    unittest.main()
