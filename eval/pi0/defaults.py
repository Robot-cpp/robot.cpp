from __future__ import annotations

from eval.common import REPO_ROOT


DEFAULT_CKPT_ROOT = REPO_ROOT / "ckpts" / "pi0-libero-finetuned-v044"
DEFAULT_GGUF_DIR = DEFAULT_CKPT_ROOT / "vlacpp-split"
DEFAULT_LEROBOT_POLICY = DEFAULT_CKPT_ROOT / "lerobot"
DEFAULT_MODEL_BASENAME = "vlacpp-pi0-libero-finetuned-v044"
DEFAULT_IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
