from dataclasses import dataclass, field
import math
import os
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers

from pointllm import conversation as conversation_lib
from pointllm.model import PointLLMLlamaForCausalLM, PointLLMConfig
from pointllm.data import make_object_point_data_module

IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    target_model_name_or_path: str = field(default="")
    student_model_name_or_path: Optional[str] = field(default=None)
    student_num_hidden_layers: int = field(default=12)


@dataclass
class DataArguments:
    data_path: str = field(default="data/objaverse_data")
    anno_path: str = field(default=None)
    use_color: bool = field(default=True)
    data_debug_num: int = field(default=0)
    split_train_val: bool = field(default=False)
    split_ratio: float = field(default=0.9)
    pointnum: int = field(default=8192)
    conversation_types: List[str] = field(default_factory=lambda: ["simple_description", "single_round", "multi_round"])
    is_multimodal: bool = True


@dataclass
class DistillArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=2048)
    remove_unused_columns: bool = field(default=False)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.0)
    warmup_ratio: float = field(default=0.03)
    max_steps: int = field(default=5000)
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=8)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=1000)
    kd_alpha: float = field(default=0.7, metadata={"help": "Weight for KD loss. CE weight = 1-kd_alpha."})
    kd_temperature: float = field(default=1.0)


def build_student_from_teacher_config(teacher_model, num_layers: int):
    teacher_cfg = teacher_model.config
    cfg_dict = teacher_cfg.to_dict()
    cfg_dict["num_hidden_layers"] = num_layers
    student_cfg = PointLLMConfig(**cfg_dict)
    student_model = PointLLMLlamaForCausalLM(student_cfg)
    return student_model


def copy_matching_weights(teacher_model, student_model):
    teacher_state = teacher_model.state_dict()
    student_state = student_model.state_dict()
    copied = 0
    for name, tensor in student_state.items():
        if name in teacher_state and teacher_state[name].shape == tensor.shape:
            tensor.copy_(teacher_state[name])
            copied += 1
    student_model.load_state_dict(student_state)
    print(f"[INFO] Copied {copied} tensors from teacher to student.")


def shift_valid_mask(labels: torch.Tensor):
    # labels: [B, L], valid prediction positions correspond to labels[:, 1:]
    return (labels[:, 1:] != IGNORE_INDEX).float()


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, DistillArguments))
    model_args, data_args, distill_args = parser.parse_args_into_dataclasses()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if distill_args.bf16 else (torch.float16 if distill_args.fp16 else torch.float32)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.target_model_name_or_path,
        cache_dir=distill_args.cache_dir,
        model_max_length=distill_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    teacher_model = PointLLMLlamaForCausalLM.from_pretrained(
        model_args.target_model_name_or_path,
        cache_dir=distill_args.cache_dir,
        torch_dtype=torch_dtype,
    ).to(device)
    teacher_model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    teacher_model.eval()
    teacher_model.requires_grad_(False)

    if model_args.student_model_name_or_path is not None:
        student_model = PointLLMLlamaForCausalLM.from_pretrained(
            model_args.student_model_name_or_path,
            cache_dir=distill_args.cache_dir,
            torch_dtype=torch_dtype,
        ).to(device)
    else:
        student_model = build_student_from_teacher_config(teacher_model, model_args.student_num_hidden_layers).to(device)
        copy_matching_weights(teacher_model, student_model)

    student_model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    student_model.train()

    point_backbone_config = teacher_model.get_model().point_backbone_config
    data_args.point_token_len = point_backbone_config["point_token_len"]
    data_args.mm_use_point_start_end = point_backbone_config["mm_use_point_start_end"]
    data_args.point_backbone_config = point_backbone_config

    data_module = make_object_point_data_module(tokenizer=tokenizer, data_args=data_args)
    train_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=distill_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=distill_args.learning_rate,
        weight_decay=distill_args.weight_decay,
    )
    total_steps = distill_args.max_steps
    warmup_steps = int(total_steps * distill_args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(distill_args.output_dir, exist_ok=True)
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(total=total_steps, desc="Draft distillation")
    while global_step < total_steps:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device) if "attention_mask" in batch else None
            point_clouds = batch.get("point_clouds", None)
            if point_clouds is not None:
                point_clouds = point_clouds.to(device=device, dtype=torch_dtype)

            with torch.no_grad():
                teacher_out = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None,
                    point_clouds=point_clouds,
                    use_cache=False,
                    return_dict=True,
                )
            student_out = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                point_clouds=point_clouds,
                use_cache=False,
                return_dict=True,
            )

            t = distill_args.kd_temperature
            teacher_logits = teacher_out.logits[:, :-1, :]
            student_logits = student_out.logits[:, :-1, :]
            valid_mask = shift_valid_mask(labels).to(student_logits.device)

            student_log_probs = F.log_softmax(student_logits / t, dim=-1)
            teacher_probs = F.softmax(teacher_logits / t, dim=-1)
            kd_token_loss = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
            kd_loss = (kd_token_loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
            kd_loss = kd_loss * (t * t)

            ce_loss = student_out.loss
            loss = distill_args.kd_alpha * kd_loss + (1.0 - distill_args.kd_alpha) * ce_loss
            loss = loss / distill_args.gradient_accumulation_steps
            loss.backward()

            if (global_step + 1) % distill_args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), distill_args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % distill_args.logging_steps == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"[step {global_step}] loss={loss.item() * distill_args.gradient_accumulation_steps:.4f}, "
                    f"ce={ce_loss.item():.4f}, kd={kd_loss.item():.4f}, lr={lr:.2e}"
                )
            if global_step % distill_args.save_steps == 0:
                ckpt_dir = os.path.join(distill_args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                student_model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
            pbar.update(1)

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    pbar.close()
    student_model.save_pretrained(distill_args.output_dir)
    tokenizer.save_pretrained(distill_args.output_dir)
    print(f"[INFO] Draft model saved to: {distill_args.output_dir}")


if __name__ == "__main__":
    main()
