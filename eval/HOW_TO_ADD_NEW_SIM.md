# How To Add A New Sim

<p align="center">
  <a href="HOW_TO_ADD_NEW_SIM_ZH.md">简体中文</a> | <strong>English</strong>
</p>

Use `eval/libero/` as the reference for simulator benchmarks. A sim integration
usually owns its environment construction, rollout loop, success metrics, and
observation adapter. It should reuse `model-server` for inference instead of
loading model weights in Python.

## Directory Shape

Recommended layout:

```text
eval/my_sim/
├── README.md
├── environment.yaml
├── policy/
│   └── model_server.py
├── runners/
│   └── run_model_server.py
├── scripts/
│   └── run_model_server.sh
└── utils/
    ├── common.py
    └── environment.py
```

Keep the Python side responsible for simulator I/O only:

- `utils/environment.py`: simulator config, env construction, reset helpers,
  task text, max episode length, success extraction.
- `policy/model_server.py`: simulator observation to `model-server` request.
- `runners/run_model_server.py`: rollout loop, result JSON, timing summaries.
- `scripts/run_model_server.sh`: optional convenience wrapper for launching the
  server and runner together.

## Model-Server Request

Every policy adapter should build this request shape:

```python
{
    "images": [
        {"name": "observation.images.image", "image": rgb_hwc_u8},
    ],
    "state": state_vector,
    "prompt": task_text,
}
```

Notes:

- Images must be RGB HWC `uint8`, or a dict accepted by
  `robot_client.python.model_client.image_to_rgb_hwc_u8_bytes`.
- Image `name` values must match the image keys stored in the GGUF metadata.
- State should be the simulator/model state in the model's expected order. Do
  not add Python-side padding unless the simulator contract explicitly requires
  it.
- `prompt` should be the task instruction used by the training/eval benchmark.

Reference: `eval/libero/policy/model_server.py`.

## Policy Adapter

Subclass `BasePolicy` when the default `RobotPolicy` cannot represent the sim
observation directly.

```python
from typing import Any

from robot_client.policy.base_policy import BasePolicy
from robot_client.python.model_client import ModelClient


class MySimModelServerPolicy(BasePolicy):
    def __init__(self, host: str, port: int, action_dim: int):
        super().__init__(ModelClient(host=host, port=port, timeout=120.0))
        self.action_dim = action_dim

    def build_observation(self, observation: dict[str, Any], *, platform: Any, task: str) -> dict[str, Any]:
        del platform
        return {
            "images": [
                {"name": "observation.images.image", "image": make_rgb_image(observation)},
            ],
            "state": make_state_vector(observation),
            "prompt": task,
        }
```

`BasePolicy.select_action()` already handles action chunk caching. Set
`action_dim` to the number of action values accepted by the simulator, so any
extra values returned by the model are ignored before `env.step()`.

## Runner Loop

A sim runner should be explicit about the benchmark lifecycle:

1. Parse runner args: host, port, suite/task ids, seed, episode count, max steps,
   image/state/action settings, output path.
2. Apply runtime environment variables before importing or constructing the sim.
3. Construct envs.
4. Wait for an existing `model-server` or launch one from `--server-command`.
5. For each episode:
   - reset the sim;
   - call `policy.reset(reset_server=True)`;
   - get the task text;
   - call `policy.select_action(observation, platform=platform, task=task)`;
   - step the sim;
   - record success, reward, steps, predict calls, and timing.
6. Write a JSON payload with config, per-episode rows, aggregate success, and
   timing summaries.
7. Close envs and stop any server process launched by the runner.

If the adapter does not use platform fields, pass a plain `BasePlatform()` stub,
as LIBERO does in `eval/libero/runners/run_model_server.py`.

## Result JSON

Keep result files easy to compare across sims:

```python
{
    "runner": "model-server",
    "config": {...},
    "episodes": [
        {
            "episode": 0,
            "seed": 1000,
            "task": "...",
            "success": True,
            "sum_reward": 1.0,
            "steps": 123,
            "predict_calls": 16,
            "server_timing_avg_ms": {...},
        }
    ],
    "timing_ms": {...},
    "overall": {...},
    "per_task": [...],
}
```

Use helpers like `write_json()` and `aggregate_episodes()` from
`eval/libero/utils/common.py` if they fit the new benchmark.

## Checklist

- Add an `environment.yaml` if the sim has nontrivial Python dependencies.
- Make image keys configurable from the runner.
- Make state and action dimensions configurable from the runner.
- Support `--host`, `--port`, `--launch-server`, and `--server-command`.
- Reset both the sim and model-server between episodes.
- Write outputs under `eval/results/` by default.
- Add a one-episode smoke command to the sim README.
- Keep generated results, videos, and checkpoints out of git.
