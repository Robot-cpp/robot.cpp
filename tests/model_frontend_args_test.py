#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys


def run(executable: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [executable, *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
    )


def require_failure(result: subprocess.CompletedProcess[str], expected: str) -> None:
    if result.returncode == 0 or expected not in result.stderr:
        raise AssertionError(
            f"expected failure containing {expected!r}; returncode={result.returncode}, "
            f"stderr={result.stderr!r}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("cli", "server"), required=True)
    parser.add_argument("--executable", required=True)
    args = parser.parse_args()

    invalid_seed = run(args.executable, "--noise-seed", "not-a-number")
    require_failure(invalid_seed, "invalid --noise-seed value 'not-a-number'")

    legacy = run(args.executable, "--model-type", "starvla_qwen_fast")
    require_failure(legacy, "unsupported model type 'starvla_qwen_fast'")

    missing_policy = run(args.executable, "--model-type", "starvla")
    require_failure(missing_policy, "starvla requires --policy")

    if args.kind == "cli":
        wrong_image_name = run(
            args.executable,
            "--model-type",
            "starvla",
            "--policy",
            "unused.gguf",
            "--image",
            "unused.png",
            "--image-name",
            "image",
        )
        require_failure(wrong_image_name, "image must be named 'image_0'")
    else:
        bad_noise_mode = run(
            args.executable,
            "--model-type",
            "starvla",
            "--policy",
            "unused.gguf",
            "--noise-mode",
            "debug-sin",
        )
        require_failure(bad_noise_mode, "does not support --noise-mode debug-sin")
    return 0


if __name__ == "__main__":
    sys.exit(main())
