# eval/ Platform Integration Guide

[中文](README_ZH.md)

The `eval/` directory holds **end-to-end evaluation and closed-loop control examples** for robot.cpp on specific robots or simulators. This code does not run model inference itself—that is handled by [`model-server`](../robot_server/README.md). `eval/` wires each platform’s observation capture, action execution, and (optionally) benchmark workflows.

The repo root [README.md](../README.md) describes the three-layer layout:

| Layer | Role | Location |
|---|---|---|
| `model-client` | Talks to `model-server` over the hand-written TCP protocol | `robot_client/python/`, `robot_client/cpp/` |
| `policy` | Converts platform observations into predict requests and caches action chunks | `robot_client/policy/` |
| `platform` | Hardware or sim adapter: sensors in, controls out | `eval/<platform>/` |

`eval/` mainly implements the **platform** layer; policy and client code live under `robot_client/` and are reused across platforms.

## Layout

```text
eval/
├── base_platform.py          # Shared base class for real-robot / sim platforms
├── libero/                   # LIBERO sim benchmark (multi-camera, batch rollout)
└── lerobot_so101/            # SO-101 real-robot sync closed-loop example
```

### Existing examples

| Directory | Scenario | Notes |
|---|---|---|
| [`libero/`](libero/README.md) | Sim eval | LIBERO benchmark with C++ policy rollout and LeRobot baseline. [中文](libero/README_ZH.md) |
| [`lerobot_so101/`](lerobot_so101/README.md) | Real robot | SO-101 follower + single-camera observe → predict → act loop. [中文](lerobot_so101/README_ZH.md) |

The two examples are organized slightly differently:

- **SO-101** follows the standard `BasePlatform` + `RobotPolicy` + `SyncControlLoop` path—use it as the template for new real-robot platforms.
- **LIBERO** implements a dedicated observation adapter under `eval/libero/policy/` (multi-camera, state packing, sim rollout) and does **not** inherit `BasePlatform`—use it as a reference for new **sim benchmarks**.

## Standard closed-loop data flow

Using SO-101 as an example, one control step looks like:

```text
Platform.get_observation()     # read camera + joint state
        ↓
Policy.build_observation()     # build model-client request (RGB, state, prompt)
        ↓
ModelClient.predict()          # TCP to model-server
        ↓
Policy.select_action()         # dequeue one step from the action chunk
        ↓
Platform.send_action()         # send to the robot
```

Relevant code:

- Control loop: [`robot_client/policy/sync_loop.py`](../robot_client/policy/sync_loop.py)
- Default policy: `RobotPolicy` in [`robot_client/policy/base_policy.py`](../robot_client/policy/base_policy.py)
- Platform base: [`eval/base_platform.py`](base_platform.py)
- SO-101 entry: [`eval/lerobot_so101/run_sync.py`](lerobot_so101/run_sync.py)

## Adding a new platform

The steps below focus on real-robot platforms. For sim benchmarks, see [`eval/libero/`](libero/README.md) and organize your own runner and policy adapter.

### step1: Create a directory

Add a subdirectory under `eval/`, e.g. `eval/my_robot/`:

```text
eval/my_robot/
├── my_robot_client.py    # BasePlatform + config_from_env + create_platform
├── run_sync.py           # entry: ModelClient + Policy + Platform + SyncControlLoop
├── shell/
│   ├── my_robot_env.sh   # serial, camera, SERVER, TASK, etc.
│   └── run_robot_client.sh
└── environment.yaml      # (optional) conda env
```

### step2: Implement a `BasePlatform` subclass

Subclass [`BasePlatform`](base_platform.py) and implement at least:

| Method / property | Required | Description |
|---|---|---|
| `connect()` / `disconnect()` | yes | Open / release serial, cameras, sim env, etc. |
| `get_observation()` | yes | Return a dict with images and joint state |
| `_send_action(action: dict)` | yes | Send `{action_key: float}` to the underlying SDK |
| `action_keys` | yes | Must match observation state fields and model action dim |
| `camera_key` | single-camera | Image field name in the local observation dict |
| `model_image_name` | single-camera | Image name sent to model-server; must match GGUF metadata |
| `on_reset_home()` | no | Platform-side motion when the user presses `R` |

`BasePlatform.send_action()` already converts numpy / list inputs to `{key: float}`; subclasses only need `_send_action()`.

Reference: [`eval/lerobot_so101/so101_client.py`](lerobot_so101/so101_client.py).

Minimal skeleton:

```python
from dataclasses import dataclass
from eval.base_platform import BasePlatform

@dataclass
class MyRobotConfig:
    port: str
    task: str
    camera_key: str = "camera1"
    model_image_name: str = "observation.images.front"
    fps: int = 25

def config_from_env() -> MyRobotConfig:
    ...

class MyRobotPlatform(BasePlatform):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_observation(self) -> dict: ...
    def _send_action(self, action: dict[str, float]) -> None: ...

    @property
    def camera_key(self) -> str:
        return self.cfg.camera_key

    @property
    def model_image_name(self) -> str:
        return self.cfg.model_image_name

    @property
    def action_keys(self) -> list[str]:
        return [...]  # populate after connect from the SDK

def create_platform(cfg: MyRobotConfig | None = None) -> MyRobotPlatform:
    return MyRobotPlatform(cfg or config_from_env())
```

### step3: Register the platform

Add an entry to `PLATFORM_MODULES` in [`eval/base_platform.py`](base_platform.py):

```python
PLATFORM_MODULES = {
    "so101": "eval.lerobot_so101.so101_client",
    "lerobot_so101": "eval.lerobot_so101.so101_client",
    "my_robot": "eval.my_robot.my_robot_client",  # new
}
```

Select the platform at runtime:

```bash
export ROBOT_PLATFORM=my_robot
```

### step4: Write the entry script

Reuse the SO-101 [`run_sync.py`](lerobot_so101/run_sync.py) pattern:

```python
from model_client import ModelClient
from eval.base_platform import create_platform
from robot_client.policy.base_policy import RobotPolicy
from robot_client.policy.sync_loop import SyncControlLoop, SyncLoopConfig

client = ModelClient(host=host, port=port)
policy = RobotPolicy(client)
platform = create_platform()

SyncControlLoop(platform, policy, SyncLoopConfig(task=cfg.task, fps=cfg.fps)).run()
```

If your observation layout is incompatible with `RobotPolicy` (multi-camera, custom state packing, extra preprocessing), subclass `BasePolicy` and override `build_observation()`. See [`eval/libero/policy/model_server.py`](libero/policy/model_server.py).

### step5: Align image keys and state dimension

This is the most common integration pitfall.

**`camera_key` vs `model_image_name`**

| Variable | Purpose |
|---|---|
| `camera_key` | Image field in the platform’s local observation dict (e.g. LeRobot `camera1`) |
| `model_image_name` | Image name sent to `model-server`; must match `smolvla.image_keys` / `pi0.image_keys` in the GGUF |

Inspect image keys in a GGUF checkpoint:

```bash
strings ckpts/<gguf_dir>/mmproj-smolvla-f32.gguf | rg "observation\.images\."
```

If the checkpoint was trained in LeRobot with a `rename_map` (e.g. `front` → `camera1`), GGUF conversion applies the **inverse** map. At runtime, use the **pre-rename** name (in this example, `observation.images.front`). See the SO-101 README for details.

**State dimension**

`RobotPolicy.build_observation()` builds the state vector by reading floats from `platform.action_keys` in order. Make sure:

- key names match the training dataset;
- order matches what the model expects;
- dimension matches the checkpoint’s `observation.state` shape.

### step6: Launch and verify

1. Start model-server (see [robot_server/README.md](../robot_server/README.md)).
2. Run your platform client shell script.
3. Confirm the server with the latency test script or [`robot_client/examples/python/minimal_example.py`](../robot_client/examples/python/minimal_example.py).
4. Run the real-robot / sim closed loop; watch the server terminal for lines like `[SmolVLA] Error:`.

SO-101 includes a camera smoke test ([`eval/lerobot_so101/test/run_camera_test.sh`](lerobot_so101/test/run_camera_test.sh)) you can adapt to validate image shape and observation encoding for your platform.

## When to write a custom policy

Default `RobotPolicy` works when:

- you have a single camera (or can split multi-camera into separate predict calls);
- state is an ordered list of float fields from the observation;
- the task string is passed to the server as `prompt`.

Subclass `BasePolicy` when:

- multiple cameras must be sent in one predict call;
- state needs transforms, normalization, or packing (e.g. LIBERO eef + gripper);
- you need LeRobot-preprocessor-equivalent rename / normalize logic.

LIBERO’s [`ModelServerPolicy`](libero/policy/model_server.py) is an example of a custom policy plus dedicated runner.

## Related docs

- [SO-101 real-robot guide](lerobot_so101/README.md) · [中文](lerobot_so101/README_ZH.md)
- [LIBERO sim eval](libero/README.md) · [中文](libero/README_ZH.md)
- [robot_server launch and protocol](../robot_server/README.md)
- [robot_client and policy](../robot_client/README.md)
- [Adding a new model runtime](../src/README.md)
