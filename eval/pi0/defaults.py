from __future__ import annotations

import os
from pathlib import Path

from eval.libero.utils import REPO_ROOT


def default_gguf_dir() -> Path:
    override = os.environ.get("VLACPP_PI0_GGUF_DIR")
    if override:
        return Path(override)
    return REPO_ROOT / "ckpts" / "pi0-libero-finetuned-v044" / "vlacpp-split"


def infer_model_basename(gguf_dir: Path) -> str:
    candidates = sorted(gguf_dir.glob("*.vit.gguf"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one *.vit.gguf in {gguf_dir}; "
            "pass --model-basename when the directory contains multiple models"
        )
    return candidates[0].name[: -len(".vit.gguf")]


DEFAULT_IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
