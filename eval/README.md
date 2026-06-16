# Eval

This tree contains evaluation entrypoints for LIBERO rollouts and real-robot SO101
control against `model-server`.

```text
eval/model_server_policy.py  reusable model-server policy, server lifecycle, and timing helpers
eval/libero/                 LIBERO environment setup, LeRobot baseline, latency, and model-server rollout runners
eval/lerobot_so101/          SO101 real-robot client, camera plugin, calibrate/teleop/record scripts
```

Start with:

- `eval/libero/README.md` for LIBERO dependencies, the LeRobot baseline, and
  model-server rollouts.
- `eval/lerobot_so101/README.md` for SO101 hardware setup and sync control loop.
