from __future__ import annotations

from eval.libero.utils import REPO_ROOT


DEFAULT_CKPT_ROOT = REPO_ROOT / "ckpts" / "pi0-libero-finetuned-v044"
DEFAULT_GGUF_DIR = DEFAULT_CKPT_ROOT / "vlacpp-split"
DEFAULT_MODEL_BASENAME = "vlacpp-pi0-libero-finetuned-v044"
DEFAULT_IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
