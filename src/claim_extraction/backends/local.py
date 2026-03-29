from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, Optional, Tuple, cast

import torch
from dotenv import dotenv_values
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.claim_extraction.config import DOTENV_PATH, ExtractionConfig

_LOCAL_MODEL_CACHE: Dict[str, Tuple[Any, Any, torch.device]] = {}


def _estimate_model_params_billions(model_name: str) -> Optional[float]:
    """Best-effort parse of model size from names like 'Qwen/Qwen3-8B'."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]\b", model_name)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _available_memory_gb(device: torch.device) -> Optional[float]:
    """Return available memory in GB for the selected execution device."""
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            return float(props.total_memory) / (1024**3)
        except Exception:
            return None

    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            kb = float(parts[1])
                            return kb / (1024**2)
        except Exception:
            return None

    return None


def _estimate_required_memory_gb(model_name: str, dtype: torch.dtype, device: torch.device) -> Optional[float]:
    """Estimate required memory for model weights plus runtime overhead."""
    params_b = _estimate_model_params_billions(model_name)
    if params_b is None:
        return None

    bytes_per_param = 2.0 if dtype in (torch.float16, torch.bfloat16) else 4.0
    weights_gb = (params_b * 1_000_000_000.0 * bytes_per_param) / (1024**3)

    # Add conservative overhead for buffers, activations and framework runtime.
    overhead_factor = 1.35 if device.type == "cuda" else 1.25
    return weights_gb * overhead_factor


def _load_local_model(model_name: str) -> Tuple[Any, Any, torch.device]:
    """Load and cache a local HF model and tokenizer."""
    if model_name in _LOCAL_MODEL_CACHE:
        return _LOCAL_MODEL_CACHE[model_name]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    hf_token = dotenv_values(DOTENV_PATH).get("HF_TOKEN")
    if hf_token:
        # Ensure downstream libraries can also see the token in this process.
        os.environ.setdefault("HF_TOKEN", hf_token)

    est_required = _estimate_required_memory_gb(model_name, dtype, device)
    available = _available_memory_gb(device)
    if est_required is not None and available is not None and est_required > available * 0.9:
        raise RuntimeError(
            (
                "Insufficient available memory for local model load: "
                f"model={model_name}, required~{est_required:.1f}GB, available~{available:.1f}GB on {device.type}. "
                "Use a smaller local model, switch to remote backend, or reduce competing memory usage."
            )
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, token=hf_token)
    model = cast(Any, model).to(device)
    _LOCAL_MODEL_CACHE[model_name] = (tokenizer, model, device)
    return tokenizer, model, device


def call_local_llm(prompt: str, config: ExtractionConfig) -> str:
    """Run local generation with transformers using chat template when available."""
    tokenizer, model, device = _load_local_model(config.model_name)

    messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        rendered = prompt

    model_inputs = tokenizer([rendered], return_tensors="pt").to(device)
    do_sample = config.temperature > 0
    generate_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = config.temperature

    generated = model.generate(**model_inputs, **generate_kwargs)
    output_ids = generated[0][len(model_inputs.input_ids[0]) :].tolist()

    # Qwen-style marker for optional reasoning blocks; keep best-effort stripping.
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    return tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
