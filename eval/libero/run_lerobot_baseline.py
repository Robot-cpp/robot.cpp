#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from eval.libero.utils import (
    DEFAULT_RESULTS_DIR,
    REPO_ROOT,
    parse_task_ids,
    task_ids_arg,
    timestamp,
    write_json,
)
from eval.libero.env import DEFAULT_LIBERO_CONFIG_PATH, apply_runtime_env, ensure_libero_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conda-env", help="optional conda env used to run lerobot-eval")
    parser.add_argument("--lerobot-eval", default="lerobot-eval")
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-ids", default="0")
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--job-name", default="libero_pi0_v044_baseline")
    parser.add_argument("--mujoco-gl", default="egl")
    parser.add_argument("--pyopengl-platform")
    parser.add_argument("--numba-cache-dir", type=Path)
    parser.add_argument("--torchinductor-cache-dir", type=Path)
    parser.add_argument("--triton-cache-dir", type=Path)
    parser.add_argument("--libero-config-path", type=Path, default=DEFAULT_LIBERO_CONFIG_PATH)
    parser.add_argument("--local-tokenizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extra-arg", action="append", default=[])
    return parser.parse_args()


def command(args: argparse.Namespace, output_dir: Path, policy_path: Path) -> list[str]:
    task_ids = task_ids_arg(parse_task_ids(args.task_ids))
    inner = [
        args.lerobot_eval,
        f"--policy.path={policy_path}",
        "--env.type=libero",
        f"--env.task={args.suite}",
        f"--eval.batch_size={args.batch_size}",
        f"--eval.n_episodes={args.n_episodes}",
        "--env.max_parallel_tasks=1",
        f"--seed={args.seed}",
        f"--output_dir={output_dir}",
        f"--job_name={args.job_name}",
    ]
    if task_ids is not None:
        inner.append(f"--env.task_ids={task_ids}")
    inner.extend(args.extra_arg)
    if args.conda_env:
        return ["conda", "run", "-n", args.conda_env, *inner]
    return inner


def prepare_policy_path(args: argparse.Namespace, output_dir: Path, env: dict[str, str]) -> Path:
    if not args.local_tokenizer:
        return args.policy_path
    source = args.policy_path.resolve()
    if not (source / "tokenizer.model").exists():
        return args.policy_path

    shadow = output_dir / "policy-local-tokenizer"
    shadow.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = shadow / item.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(item)

    preprocessor = json.loads((source / "policy_preprocessor.json").read_text(encoding="utf-8"))
    for step in preprocessor.get("steps", []):
        if step.get("registry_name") == "tokenizer_processor":
            step.setdefault("config", {})["tokenizer_name"] = str(shadow)
    (shadow / "policy_preprocessor.json").unlink(missing_ok=True)
    (shadow / "policy_preprocessor.json").write_text(json.dumps(preprocessor, indent=2) + "\n", encoding="utf-8")
    (shadow / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "GemmaTokenizer",
                "model_type": "gemma",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (shadow / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    return shadow


def prepare_libero_config(args: argparse.Namespace, env: dict[str, str]) -> None:
    config_path = args.libero_config_path.expanduser()
    env["LIBERO_CONFIG_PATH"] = str(config_path)
    if args.conda_env:
        script = (
            "import sys; "
            f"sys.path.insert(0, {str(REPO_ROOT)!r}); "
            "from pathlib import Path; "
            "from eval.libero.env import ensure_libero_config; "
            "import os; "
            "ensure_libero_config(Path(os.environ['LIBERO_CONFIG_PATH']))"
        )
        proc = subprocess.run(
            ["conda", "run", "-n", args.conda_env, "python", "-c", script],
            env=env,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            if proc.stdout:
                print(proc.stdout)
            if proc.stderr:
                print(proc.stderr, file=sys.stderr)
            raise RuntimeError(f"failed to prepare LIBERO config at {config_path}")
        return
    ensure_libero_config(config_path)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or DEFAULT_RESULTS_DIR / f"lerobot-baseline-{timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    apply_runtime_env(args, env)
    policy_path = prepare_policy_path(args, output_dir, env)
    prepare_libero_config(args, env)
    cmd = command(args, output_dir, policy_path)

    print("Running LeRobot baseline:")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
    (output_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    report = {
        "runner": "lerobot-eval",
        "returncode": proc.returncode,
        "command": cmd,
        "output_dir": str(output_dir),
        "eval_info": str(output_dir / "eval_info.json"),
        "libero_config_path": env["LIBERO_CONFIG_PATH"],
        "policy_path": str(policy_path),
        "env": {
            "MUJOCO_GL": env.get("MUJOCO_GL"),
            "PYOPENGL_PLATFORM": env.get("PYOPENGL_PLATFORM"),
            "NUMBA_CACHE_DIR": env.get("NUMBA_CACHE_DIR"),
            "TORCHINDUCTOR_CACHE_DIR": env.get("TORCHINDUCTOR_CACHE_DIR"),
            "TRITON_CACHE_DIR": env.get("TRITON_CACHE_DIR"),
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": env.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"),
        },
    }
    write_json(output_dir / "baseline_run.json", report)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode == 0:
        print(f"wrote {output_dir / 'eval_info.json'}")
    else:
        print(f"LeRobot baseline failed; logs are in {output_dir}", file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
