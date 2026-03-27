from typing import Optional

import torch
from transformers import AutoModelForCausalLM


def load_assistant_model(assistant_model_name: Optional[str], torch_dtype: torch.dtype):
    """Load a draft model for speculative decoding.

    Returns None when assistant_model_name is not provided.
    """
    if not assistant_model_name:
        return None

    assistant_model = AutoModelForCausalLM.from_pretrained(
        assistant_model_name,
        low_cpu_mem_usage=True,
        use_cache=True,
        torch_dtype=torch_dtype,
    ).cuda()
    assistant_model.eval()
    return assistant_model


def generate_with_speculative_fallback(
    model,
    input_ids,
    point_clouds,
    stopping_criteria,
    do_sample=True,
    temperature=1.0,
    top_k=50,
    max_length=2048,
    top_p=0.95,
    assistant_model=None,
    num_assistant_tokens: int = 8,
    assistant_confidence_threshold: float = 0.4,
):
    """Generate outputs with speculative decoding when possible.

    Falls back to vanilla generation when assistant generation is unsupported.
    """
    generate_kwargs = {
        "do_sample": do_sample,
        "temperature": temperature,
        "top_k": top_k,
        "max_length": max_length,
        "top_p": top_p,
        "stopping_criteria": [stopping_criteria],
    }

    if assistant_model is None:
        return model.generate(input_ids, point_clouds=point_clouds, **generate_kwargs)

    try:
        return model.generate(
            input_ids,
            point_clouds=point_clouds,
            assistant_model=assistant_model,
            num_assistant_tokens=num_assistant_tokens,
            assistant_confidence_threshold=assistant_confidence_threshold,
            **generate_kwargs,
        )
    except TypeError as exc:
        if "assistant_model" not in str(exc) and "num_assistant_tokens" not in str(exc):
            raise
        print("[Warning] Current transformers version does not support assistant generation kwargs. Falling back to standard decoding.")
        return model.generate(input_ids, point_clouds=point_clouds, **generate_kwargs)
