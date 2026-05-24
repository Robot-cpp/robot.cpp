# vla.cpp LIBERO Timing Analysis

This note summarizes saved LIBERO timing JSON files from earlier vla.cpp and
LeRobot runs. No new simulator rollouts were run for this note.

## Measurement Setup

- Task suite: `libero_object`
- Episodes: 10 tasks x 5 episodes for all-task rows
- Flow steps: 10
- Action chunk size used by the environment runner: 50 actions
- Timing prefix discard: first chunk is removed from steady-state chunk timing
- Hardware used by the report: Intel Xeon Platinum 8358P and NVIDIA
  A100-PCIE-40GB

## Main Results

| Runtime | Device | Success | Chunk infer (s) | Chunk policy e2e (s) | Step policy e2e (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| vla.cpp | CUDA | 41/50 | 0.145 | 0.149 | 0.007 |
| vla.cpp | CUDA | 39/50 | 0.142 | 0.147 | 0.007 |
| vla.cpp | CPU | 37/50 | 5.915 | 5.919 | 0.136 |
| LeRobot | CUDA, `compile_model=False` | 42/50 | - | 0.235 | 0.008 |
| LeRobot | CUDA, `compile_model=True` | 40/50 | - | 0.096 | 0.009 |
| LeRobot | CPU, `compile_model=False` | 1/1 | - | 13.666 | 0.334 |

## vla.cpp Stage Breakdown

The newer vla.cpp JSON files split policy-side work into preprocessing, chunk
input preparation, noise generation, chunk inference, and postprocessing.

| Run | Success | Preprocess (s) | Prepare (s) | Noise (s) | Chunk infer (s) | Postprocess (s) | Chunk e2e (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CUDA all tasks | 41/50 | 0.0019 | 0.0022 | 0.0010 | 0.1446 | 0.00003 | 0.1491 |
| CUDA all tasks, offset 5 | 39/50 | 0.0022 | 0.0023 | 0.0011 | 0.1419 | 0.00003 | 0.1470 |
| CPU all tasks | 37/50 | 0.0022 | 0.0023 | 0.0001 | 5.9148 | 0.00003 | 5.9192 |

The CUDA overhead around inference is small: chunk policy e2e is about
4-5 ms higher than chunk infer. On CPU, chunk e2e is almost entirely model
inference.

## Component-Level Timing

The saved JSON files do not contain direct VLM, ViT, or action expert timers.
The most detailed reliable fields from the LIBERO runner are the stage fields
above: preprocess, prepare, noise, chunk infer, and postprocess. In these files,
`chunk_infer` is the whole model-side action chunk generation call, so it should
not be reported as separate ViT/VLM/action-expert time.

There are older single-task profiling and ablation records that are useful as
engineering references, but they are not additive component timings:

| Artifact | Scope | Steady-state timing |
| --- | --- | ---: |
| `/tmp/vlacpp-profile-cuda-mtmd-object-task0-n1-deviceprefix.json` | full policy profile, device prefix | policy 0.172 s |
| `/tmp/vlacpp-profile-cuda-mtmd-object-task0-n1-deviceprefix-nographs.json` | full policy profile, no CUDA graphs | policy 0.187 s |
| `/tmp/vlacpp-profile-cuda-mtmd-object-task0-n1-prefixcache.json` | full policy profile, prefix cache variant | policy 0.185 s |
| `/tmp/vlacpp-profile-cuda-mtmd-object-task0-n1.json` | older full policy profile | policy 0.230 s |
| `/tmp/vlacpp-task1-n1-baseonly-cuda.json` | single-task base image-set run | chunk infer 0.116 s |
| `/tmp/vlacpp-profile-task1-skip-prefix-output.json` | single-task skip-prefix-output run | chunk infer 0.126 s |
| `/tmp/vlacpp-task1-n1-cublas16-cuda.json` | single-task cuBLAS16 run | chunk infer 0.125 s |
| `/tmp/vlacpp-task1-n1-mtmd-no-extra-copy-cuda.json` | single-task MTMD copy-path run | chunk infer 0.148 s |

These numbers should not be interpreted as "ViT = X ms, VLM = Y ms, action
expert = Z ms." Real component timing would need explicit timers around vision
encoding, language/prefix prefill, the action denoising loop, and final action
projection in the runtime.

## Timing Fields

- `chunk_infer_time_excluding_prefix_s`: vla.cpp-only `policy.infer(...)` time
  after dropping the first chunk.
- `chunk_policy_e2e_time_excluding_prefix_s`: chunk-generation policy path after
  dropping the first chunk. For vla.cpp this includes preprocessing, input
  preparation, noise generation, and `policy.infer(...)`. For LeRobot this is
  the timed `policy.select_action(...)` call on chunk-generation steps.
- `step_policy_e2e_time_s`: policy-side work per simulator control step. Most
  steps reuse cached actions and do not generate a new chunk.

Older LeRobot JSON files only contain chunk/step e2e fields. They do not split
preprocess, postprocess, or model-forward time.

## Source JSON Files

- `/tmp/vlacpp-lerobot-env-alltasks-n5-currentcode-cuda.json`
- `/tmp/vlacpp-lerobot-env-alltasks-n5-offset5-currentcode-cuda.json`
- `/tmp/vlacpp-lerobot-env-alltasks-n5-cpu-backend-cache.json`
- `/tmp/lerobot-native-controlled-noise-alltasks-n5.json`
- `/tmp/lerobot-native-controlled-noise-alltasks-n5-compileconfig-current.json`
- `/tmp/lerobot-native-task1-n1-cpu-controlled.json`

The table can be regenerated locally with:

```sh
python3 eval/summarize-libero-timing.py \
  /tmp/vlacpp-lerobot-env-alltasks-n5-currentcode-cuda.json \
  /tmp/vlacpp-lerobot-env-alltasks-n5-offset5-currentcode-cuda.json \
  /tmp/vlacpp-lerobot-env-alltasks-n5-cpu-backend-cache.json \
  /tmp/lerobot-native-controlled-noise-alltasks-n5.json \
  /tmp/lerobot-native-controlled-noise-alltasks-n5-compileconfig-current.json \
  /tmp/lerobot-native-task1-n1-cpu-controlled.json
```
