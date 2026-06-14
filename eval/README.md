# Eval

This tree contains the LIBERO eval entrypoints used for pi0 model-server
rollouts and LeRobot reference runs.

```text
eval/model_server_policy.py Generic model-server action-chunk adapter
eval/libero/                LIBERO env setup, request builders, LeRobot baseline
eval/pi0/                   pi0 model-server LIBERO runner
```

Start with:

- `eval/libero/README.md` for LIBERO dependencies and the LeRobot baseline.
- `eval/pi0/README.md` for downloading split GGUF files and running pi0 through
  `model-server`.
