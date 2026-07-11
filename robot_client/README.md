<p align="center">
  <a href="README_ZH.md">简体中文</a> | <strong>English</strong>
</p>

# Robot Client

This directory provides client-side code for the vla.cpp `model-server`. The client is only responsible for encoding the current observation as a TCP request, sending it to the server, and parsing the returned action chunk and timing information. Robot observation collection and action execution are handled by the upper-level platform / policy code.

## Usage

### step0: Start model-server first

All clients require the vla.cpp server to be running. The most common address is:

```text
127.0.0.1:5555
```

For example, from the repository root:

```bash
bash robot_server/shell/launch_robot_server_mac_cpu.sh
```

If you are using Windows, CUDA, or a different model checkpoint, use the corresponding script under `robot_server/shell/`. The only requirement is that the host/port match the client configuration.

### step1: Minimal Python smoke test

The Python TCP client is located at:

```text
robot_client/python/model_client.py
```

It provides:

* `health()`: check whether the server is available
* `reset()`: reset server-side cache/state
* `shutdown()`: ask the server to exit
* `predict(observation)`: send an observation and return an action chunk

The minimal Python example is located at:

```text
robot_client/examples/python/minimal_example.py
```

After starting the server, run:

```bash
python robot_client/examples/python/minimal_example.py
```

You can also run a benchmark smoke test through the wrapper script:

```bash
export VLA_CPP_ROOT=/path/to/vla.cpp/robot.cpp
bash robot_client/shell/client_example.sh
```

Note:

* `client_example.sh` depends on the `VLA_CPP_ROOT` environment variable.
* `minimal_example.py` connects to `127.0.0.1:5555` by default.
* An observation must contain at least `images`, `state`, and `prompt`.

### step2: Python observation format

The Python client accepts an observation as a dict:

```python
{
    "images": [
        {
            "name": "image",
            "image": image_hwc_uint8,
        }
    ],
    "state": state_vector,
    "prompt": "grab the block.",
}
```

Images can be passed directly through `image`, or through pre-packed raw RGB fields:

```python
{
    "name": "observation.images.camera1",
    "rgb_hwc_u8": rgb_bytes,
    "width": 224,
    "height": 224,
    "stride_bytes": 224 * 3,
}
```

`model_client.py` normalizes the input into contiguous `RGB / HWC / uint8` bytes and sends it to the server using the vla.cpp TCP protocol.

### step3: Read the response

`ModelClient.predict()` returns a `ModelResponse`:

| Field | Description |
| --- | --- |
| `chunk_size` | Number of action steps returned by the server in one response |
| `action_dim` | Dimension of each action step |
| `actions_flat` | Flat action buffer with length `chunk_size * action_dim` |
| `actions` | 2D list with shape `[chunk_size][action_dim]` |
| `timings` | Per-stage timings returned by the server, such as `vision_ms`, `llm_ms`, and `model_total_ms` |

A typical usage pattern is to push `response.actions` into a queue and pop one action row per control-loop step:

```python
response = client.predict(observation)
first_action = response.actions[0]
```

### step4: Real-robot policy / sync loop

The real-robot synchronous control loop uses `robot_client/policy`:

```text
robot_client/policy/base_policy.py   # BasePolicy / RobotPolicy
robot_client/policy/sync_loop.py     # observe -> select_action -> send_action
robot_client/policy/sim_policy.py    # Policy helper for simulation evaluation
```

Where:

* `BasePolicy` manages `ModelClient`, the action queue, and `select_action`
* `RobotPolicy` converts platform observations into model-server observations
* `SyncControlLoop` connects the platform and policy, and runs them at the target FPS
* `SimPolicy` reuses the server lifecycle and timing statistics for simulation evaluations such as LIBERO

The SO101 real-robot entry point is:

```bash
bash eval/lerobot_so101/script/shell/run_robot_client.sh
```

### step5: C++ client

The C++ client is located at:

```text
robot_client/cpp/model_client.h
robot_client/cpp/model_client.cpp
```

The minimal C++ example is located at:

```text
robot_client/examples/cpp/minimal_example.cpp
```

After starting the server, run:

```bash
export VLA_CPP_ROOT=/path/to/vla.cpp/robot.cpp
bash robot_client/shell/cpp_client_example.sh
```

Configurable variables:

| Variable | Description |
| --- | --- |
| `VLA_CPP_ROOT` | Repository root, required by the script |
| `BUILD_DIR` | CMake build directory |
| `HOST` / `PORT` | model-server address |
| `BUILD_CLIENT` | Set to `1` to force re-configuring / rebuilding the C++ example |
| `CMAKE_BIN` | CMake executable |

## Current Implementation

Directory layout:

```text
robot_client/
├── python/model_client.py              # Python TCP client and protocol codec
├── cpp/model_client.{h,cpp}            # C++ TCP client
├── examples/python/minimal_example.py  # Minimal Python request example
├── examples/cpp/minimal_example.cpp    # Minimal C++ request example
├── shell/client_example.sh             # Python smoke-test wrapper
├── shell/cpp_client_example.sh         # C++ example build/run wrapper
└── policy/
    ├── base_policy.py                  # BasePolicy / RobotPolicy
    ├── sim_policy.py                   # Simulation evaluation helper
    └── sync_loop.py                    # Synchronous control loop
```

The overall call chain is:

```text
platform.get_observation()
  -> policy.build_observation()
  -> ModelClient.predict()
  -> policy.select_action()
  -> platform.send_action()
```

For real robots, `platform.send_action()` converts the model output vector into `{joint_name: value}` according to `action_keys`. For simulation, the numpy action is usually passed directly to `env.step()`.
