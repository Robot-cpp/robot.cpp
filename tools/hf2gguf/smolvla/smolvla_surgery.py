#!/usr/bin/env python3
"""
SmolVLA Surgery Script
Extract components from SmolVLA model checkpoint into separate PyTorch files for GGUF conversion.

Outputs:
  - smolvla.vision.pt         (SigLIP vision encoder)
  - smolvla.connector.pt      (vision → language projector)
  - smolvla.text_model.pt     (VLM backbone, 16 layers)
  - smolvla.state_proj.pt     (state projector)
  - smolvla.action_expert.pt  (Action Expert transformer + action projections)
"""
import argparse
import torch
import os
import json
import shutil
from pathlib import Path
from safetensors import safe_open


VISION_PREFIX = "model.vlm_with_expert.vlm.model.vision_model."
CONNECTOR_PREFIX = "model.vlm_with_expert.vlm.model.connector."
TEXT_MODEL_PREFIX = "model.vlm_with_expert.vlm.model.text_model."
LM_HEAD_PREFIX = "model.vlm_with_expert.vlm.lm_head."
LM_EXPERT_PREFIX = "model.vlm_with_expert.lm_expert."
STATE_PROJ_PREFIX = "model.state_proj."

ACTION_PROJ_NAMES = (
    "action_in_proj",
    "action_out_proj",
    "action_time_mlp",
)

DEFAULT_VLM_MODEL_NAME = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VLM_FILES = (
    ("config.json", "vlm_config.json"),
    ("tokenizer.json", "tokenizer.json"),
    ("tokenizer_config.json", "tokenizer_config.json"),
    ("vocab.json", "vocab.json"),
    ("merges.txt", "merges.txt"),
    ("special_tokens_map.json", "special_tokens_map.json"),
    ("added_tokens.json", "added_tokens.json"),
    ("generation_config.json", "generation_config.json"),
    ("chat_template.json", "chat_template.json"),
)

#TODO: actually I do not test downloading from Hugging Face, try this later
def _download_vlm_file(repo_id: str, filename: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            f"VLM file is not a local path and huggingface_hub is unavailable: {exc}"
        ) from exc

    try:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename))
    except Exception as exc:
        raise RuntimeError(f"Failed to download {filename} from Hugging Face repo '{repo_id}': {exc}") from exc


def _resolve_vlm_file(vlm_model_name: str, filename: str) -> Path:
    model_ref = Path(vlm_model_name).expanduser()
    if model_ref.exists():
        vlm_file = model_ref / filename
        if not vlm_file.exists():
            raise FileNotFoundError(f"Local VLM path has no {filename}: {model_ref}")
        return vlm_file

    try:
        return _download_vlm_file(vlm_model_name, filename)
    except RuntimeError as first_error:
        if vlm_model_name == DEFAULT_VLM_MODEL_NAME:
            raise
        print(f"   Warning: could not load VLM {filename} from '{vlm_model_name}': {first_error}")
        print(f"   Falling back to {DEFAULT_VLM_MODEL_NAME}")
        return _download_vlm_file(DEFAULT_VLM_MODEL_NAME, filename)


def _copy_vlm_files(config: dict, output_dir: Path) -> None:
    vlm_model_name = str(config.get("vlm_model_name") or DEFAULT_VLM_MODEL_NAME)
    copied_files = {}

    for src_name, dst_name in VLM_FILES:
        src = _resolve_vlm_file(vlm_model_name, src_name)
        shutil.copy(src, output_dir / dst_name)
        copied_files[src_name] = src

    vlm_config_file = output_dir / "vlm_config.json"
    with open(vlm_config_file, "r", encoding="utf-8") as f:
        vlm_config = json.load(f)
    if "vision_config" not in vlm_config or "text_config" not in vlm_config:
        raise RuntimeError(f"VLM config must contain vision_config and text_config: {vlm_config_file}")

    print(f"📋 Copied vlm_config.json from {copied_files['config.json']}")
    print(f"🔤 Copied tokenizer files from {vlm_model_name}")


def extract_smolvla_components(model_path, output_dir):
    """Extract all components from SmolVLA model"""
    
    print(f"🏥 Starting SmolVLA Surgery on: {model_path}")
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find the pretrained_model directory
    pretrained_path = model_path / "pretrained_model"
    if not pretrained_path.exists():
        pretrained_path = model_path
    
    # Find safetensors files
    safetensors_file = pretrained_path / "model.safetensors"
    if safetensors_file.exists():
        safetensors_files = [safetensors_file]
    else:
        safetensors_files = sorted(pretrained_path.glob("model-*.safetensors"))
        if not safetensors_files:
            raise ValueError(f"No safetensors files found in {pretrained_path}")
    
    print(f"📂 Loading model from {len(safetensors_files)} safetensors file(s)")
    for path in safetensors_files:
        print(f"   - {path.name}")
    
    # Storage for extracted components
    vision_state_dict = {}
    connector_state_dict = {}
    text_model_state_dict = {}
    lm_expert_state_dict = {}
    state_proj_state_dict = {}
    action_projs_state_dict = {}
    lm_head_state_dict = {}
    
    # Process safetensors files
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                
                # Vision model (SigLIP)
                if key.startswith(VISION_PREFIX):
                    clean_key = key.replace(VISION_PREFIX, "")
                    vision_state_dict[clean_key] = tensor
                
                # Connector (mm_projector)
                elif key.startswith(CONNECTOR_PREFIX):
                    clean_key = key.replace(CONNECTOR_PREFIX, "")
                    connector_state_dict[clean_key] = tensor
                
                # Text model (LLM backbone)
                elif key.startswith(TEXT_MODEL_PREFIX):
                    clean_key = key.replace(TEXT_MODEL_PREFIX, "")
                    text_model_state_dict[clean_key] = tensor
                
                # LM head
                elif key.startswith(LM_HEAD_PREFIX):
                    clean_key = key.replace(LM_HEAD_PREFIX, "")
                    lm_head_state_dict[clean_key] = tensor
                
                # LM Expert (Action Expert)
                elif key.startswith(LM_EXPERT_PREFIX):
                    clean_key = key.replace(LM_EXPERT_PREFIX, "")
                    lm_expert_state_dict[clean_key] = tensor
                
                # State projector (keep original float32 precision)
                elif key.startswith(STATE_PROJ_PREFIX):
                    clean_key = key.replace(STATE_PROJ_PREFIX, "")
                    state_proj_state_dict[clean_key] = tensor
                
                # Action projections and time MLP
                elif any(name in key for name in ACTION_PROJ_NAMES):
                    clean_key = key.replace("model.", "")
                    action_projs_state_dict[clean_key] = tensor
    
    # Print extraction statistics
    print(f"\n📊 Extraction Statistics:")
    print(f"   Vision model:    {len(vision_state_dict):4d} tensors")
    print(f"   Connector:       {len(connector_state_dict):4d} tensors")
    print(f"   Text model:      {len(text_model_state_dict):4d} tensors")
    print(f"   LM head:         {len(lm_head_state_dict):4d} tensors")
    print(f"   LM Expert:       {len(lm_expert_state_dict):4d} tensors")
    print(f"   State proj:      {len(state_proj_state_dict):4d} tensors")
    print(f"   Action projs:    {len(action_projs_state_dict):4d} tensors")
    
    # Copy config files
    config_file = pretrained_path / "config.json"
    if config_file.exists():
        shutil.copy(config_file, output_dir / "config.json")
        print(f"\n📋 Copied config.json")
        
        # Also load and print key config values
        with open(config_file, encoding="utf-8") as f:
            config = json.load(f)
        print(f"   - chunk_size: {config.get('chunk_size', 'N/A')}")
        print(f"   - max_state_dim: {config.get('max_state_dim', 'N/A')}")
        print(f"   - max_action_dim: {config.get('max_action_dim', 'N/A')}")
        print(f"   - num_steps: {config.get('num_steps', 'N/A')}")
        print(f"   - vlm_model_name: {config.get('vlm_model_name', 'N/A')}")
        _copy_vlm_files(config, output_dir)
    else:
        raise RuntimeError(f"Missing required config.json: {config_file}")
    
    # Copy normalizer stats (for denormalization during inference)
    normalizer_file = pretrained_path / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    if normalizer_file.exists():
        shutil.copy(normalizer_file, output_dir / "normalizer_stats.safetensors")
        print(f"📊 Copied normalizer stats")
    
    unnormalizer_file = pretrained_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    if unnormalizer_file.exists():
        shutil.copy(unnormalizer_file, output_dir / "unnormalizer_stats.safetensors")
        print(f"📊 Copied unnormalizer stats")
    
    # Save all components
    print(f"\n💾 Saving components to {output_dir}...")
    
    # 1. Vision model
    vision_output = output_dir / "smolvla.vision.pt"
    print(f"   [1/5] Vision model -> {vision_output.name}")
    torch.save(vision_state_dict, vision_output)
    
    # 2. Connector
    connector_output = output_dir / "smolvla.connector.pt"
    print(f"   [2/5] Connector -> {connector_output.name}")
    torch.save(connector_state_dict, connector_output)
    
    # 3. Text model (LLM)
    text_model_output = output_dir / "smolvla.text_model.pt"
    print(f"   [3/5] Text model -> {text_model_output.name}")
    # Include lm_head in text_model for consistency
    text_model_state_dict["lm_head.weight"] = lm_head_state_dict.get("weight")
    torch.save(text_model_state_dict, text_model_output)
    
    # 4. State projector
    state_proj_output = output_dir / "smolvla.state_proj.pt"
    print(f"   [4/5] State proj -> {state_proj_output.name}")
    torch.save(state_proj_state_dict, state_proj_output)
    
    # 5. Action Expert (lm_expert + action projections)
    action_expert_merged = {}
    for k, v in lm_expert_state_dict.items():
        action_expert_merged[f"lm_expert.{k}"] = v
    for k, v in action_projs_state_dict.items():
        action_expert_merged[k] = v
    
    action_expert_output = output_dir / "smolvla.action_expert.pt"
    print(f"   [5/5] Action Expert -> {action_expert_output.name}")
    torch.save(action_expert_merged, action_expert_output)
    
    print(f"\n✅ SmolVLA Surgery Complete!")
    print(f"📂 Output directory: {output_dir}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Extract SmolVLA components for GGUF conversion"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to SmolVLA model directory"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for surgery results"
    )
    
    args = parser.parse_args()
    extract_smolvla_components(args.model, args.output_dir)
    
    print("\n🔧 Next steps:")
    print("   1. Run convert_smolvla_vision_to_gguf.py for Vision + Connector")
    print("   2. Run convert_smolvla_state_proj_to_gguf.py for State Projector")
    print("   3. Run convert_smolvla_action_expert_to_gguf.py for Action Expert")
    print("   4. Run convert_smolvla_llm_to_gguf.py for LLM backbone")


if __name__ == "__main__":
    main()
