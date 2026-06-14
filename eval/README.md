# Eval

The eval tree is split by responsibility:

```text
eval/common.py              shared result and path helpers
eval/libero/                LIBERO env setup, LeRobot baseline, result compare
eval/pi0/                   pi0 model-server policy and runners
```

Top-level scripts such as `eval/run_lerobot_baseline.py` are thin compatibility
wrappers around the package modules.
