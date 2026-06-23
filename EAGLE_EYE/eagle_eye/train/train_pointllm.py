import argparse
import json
import os
import sys
from typing import Any, Dict, List

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup

from eagle_eye.model.cnets import Model
from eagle_eye.model.configs import EConfig


def add_pointllm_to_path(pointllm_repo_path):
    if pointllm_repo_path and pointllm_repo_path not in sys.path:
        sys.path.insert(0, pointllm_repo_path)


def list_files(path):
    datapath = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".ckpt") or file.endswith(".pt"):
                datapath.append(os.path.join(root, file))
    return sorted(datapath)


def load_tensor_from_safetensors(path, key):
    with safe_open(path, framework="pt", device="cpu") as f:
        tensor_slice = f.get_slice(key)
        shape = tensor_slice.get_shape()
        return tensor_slice[:, : shape[1]].float()


def load_weight_from_checkpoint(model_path, key):
    safetensors_index = os.path.join(model_path, "model.safetensors.index.json")
    bin_index = os.path.join(model_path, "pytorch_model.bin.index.json")
    single_safetensors = os.path.join(model_path, "model.safetensors")
    single_bin = os.path.join(model_path, "pytorch_model.bin")

    if os.path.exists(safetensors_index):
        with open(safetensors_index, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        return load_tensor_from_safetensors(os.path.join(model_path, weight_map[key]), key)

    if os.path.exists(single_safetensors):
        return load_tensor_from_safetensors(single_safetensors, key)

    if os.path.exists(bin_index):
        with open(bin_index, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        weights = torch.load(os.path.join(model_path, weight_map[key]), map_location="cpu")
        return weights[key].float()

    if os.path.exists(single_bin):
        weights = torch.load(single_bin, map_location="cpu")
        return weights[key].float()

    raise FileNotFoundError(f"Cannot find checkpoint tensor {key} under {model_path}")


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        data["hidden_state_big"] = tensor + noise
        return data


class CustomDataset(Dataset):
    def __init__(self, datapath, max_len, transform=None):
        self.data = datapath
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.load(self.data[index], map_location="cpu")
        hidden_state = data["hidden_state"][: self.max_len][None, :]
        inputs_embeds = data["inputs_embeds"][: self.max_len][None, :]
        input_ids = data["input_ids"][: self.max_len][None, :]
        loss_mask = data["loss_mask"][: self.max_len][None, :]

        length = hidden_state.shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        input_ids_target = torch.cat((input_ids[:, 1:], torch.zeros(1, 1, dtype=input_ids.dtype)), dim=1)
        inputs_embeds_target = torch.cat(
            (
                inputs_embeds[:, 1:, :],
                torch.zeros(1, 1, inputs_embeds.shape[2], dtype=inputs_embeds.dtype),
            ),
            dim=1,
        )
        target = torch.cat(
            (
                hidden_state[:, 1:, :],
                torch.zeros(1, 1, hidden_state.shape[2], dtype=hidden_state.dtype),
            ),
            dim=1,
        )

        compression = data.get("point_token_compression", {})
        coverage_score = float(
            compression.get(
                "coverage_score",
                compression.get("coverage_cell_ratio", 1.0),
            )
        )
        new_data = {
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "target": target,
            "hidden_state_big": hidden_state,
            "input_ids": input_ids_target,
            "inputs_embeds": inputs_embeds_target,
            "point_coverage_score": coverage_score,
        }
        if self.transform:
            new_data = self.transform(new_data)
        return new_data




def _batch_normalized(score, eps=1e-6):
    score = score.float()
    mean = score.detach().mean().clamp_min(eps)
    return (score / mean).clamp(0.5, 2.0)


def build_point_rollout_factor(args, data, token_weight, visual_token_lookup, target_p):
    if not args.point_rollout_adaptive:
        return None

    token_weight = token_weight.float()
    denom = token_weight.sum(dim=1).clamp_min(1e-5)

    # Long answers carry more compounding draft errors, so rollout is emphasized there.
    length_score = _batch_normalized(denom.sqrt())

    # Teacher confidence proxies visual/text uncertainty without needing extra labels.
    teacher_conf = target_p.max(dim=2).values.detach()
    uncertainty = ((1.0 - teacher_conf) * token_weight).sum(dim=1) / denom
    uncertainty_score = _batch_normalized(uncertainty)

    # Point-cloud ambiguity proxy: compressed visual token embeddings with higher variance
    # usually correspond to more heterogeneous scenes or harder token selection.
    visual_score = torch.ones_like(length_score)
    if "input_ids" in data and "inputs_embeds" in data:
        ids = data["input_ids"].to(token_weight.device).clamp(min=0, max=visual_token_lookup.numel() - 1)
        visual_mask = visual_token_lookup[ids].bool()
        embeds = data["inputs_embeds"].to(token_weight.device).float()
        scores = []
        for batch_idx in range(embeds.shape[0]):
            cur = embeds[batch_idx][visual_mask[batch_idx]]
            if cur.shape[0] <= 1:
                scores.append(embeds.new_tensor(1.0))
            else:
                center = cur.mean(dim=0, keepdim=True)
                scores.append((cur - center).pow(2).mean().sqrt())
        visual_score = _batch_normalized(torch.stack(scores).to(token_weight.device))

    coverage_score = torch.ones_like(length_score)
    if args.rollout_coverage_weight > 0 and "point_coverage_score" in data:
        coverage = data["point_coverage_score"].to(token_weight.device).float().clamp(0.0, 1.0)
        coverage_difficulty = (1.0 - coverage).clamp_min(1e-4)
        coverage_score = _batch_normalized(coverage_difficulty)

    factor = (
        1.0
        + args.rollout_length_weight * (length_score - 1.0)
        + args.rollout_uncertainty_weight * (uncertainty_score - 1.0)
        + args.rollout_visual_weight * (visual_score - 1.0)
        + args.rollout_coverage_weight * (coverage_score - 1.0)
    )
    return factor.clamp(args.rollout_adaptive_min, args.rollout_adaptive_max).detach()


class DataCollatorWithPadding:
    @staticmethod
    def paddingtensor(intensors, length):
        _, n, dim = intensors.shape
        return torch.cat(
            (intensors, torch.zeros(1, length - n, dim, dtype=intensors.dtype)),
            dim=1,
        )

    @staticmethod
    def paddingtensor2d(intensors, length):
        _, n = intensors.shape
        return torch.cat((intensors, torch.zeros(1, length - n, dtype=intensors.dtype)), dim=1)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["hidden_state_big"].shape[1] for item in features)
        return {
            "input_ids": torch.cat([self.paddingtensor2d(item["input_ids"], max_length) for item in features]),
            "inputs_embeds": torch.cat([self.paddingtensor(item["inputs_embeds"], max_length) for item in features]),
            "hidden_states": torch.cat([self.paddingtensor(item["hidden_state_big"], max_length) for item in features]),
            "target": torch.cat([self.paddingtensor(item["target"], max_length) for item in features]),
            "loss_mask": torch.tensor(
                [item["loss_mask"] + [0] * (max_length - len(item["loss_mask"])) for item in features]
            ),
            "attention_mask": torch.tensor(
                [item["attention_mask"] + [0] * (max_length - len(item["attention_mask"])) for item in features]
            ),
            "point_coverage_score": torch.tensor(
                [float(item.get("point_coverage_score", 1.0)) for item in features], dtype=torch.float32
            ),
        }


POINT_VISUAL_KEYWORDS = (
    "object", "shape", "color", "colour", "red", "blue", "green", "yellow", "black", "white",
    "brown", "gray", "grey", "orange", "purple", "pink", "round", "rect", "square", "cube",
    "cubic", "cylinder", "cylind", "sphere", "spherical", "cone", "flat", "thin", "thick",
    "long", "short", "wide", "narrow", "tall", "small", "large", "front", "back", "top",
    "bottom", "side", "left", "right", "under", "over", "above", "below", "beside", "inside",
    "outside", "attached", "connected", "part", "handle", "wheel", "leg", "arm", "seat", "chair",
    "table", "car", "truck", "bus", "train", "airplane", "plane", "boat", "house", "roof",
    "door", "window", "body", "head", "tail", "wing", "base", "stand", "pole", "box", "toy",
    "cartoon", "wood", "wooden", "metal", "plastic", "glass", "material", "texture", "surface",
)


def build_visual_token_lookup(tokenizer, vocab_size, extra_keywords=None):
    keywords = list(POINT_VISUAL_KEYWORDS)
    if extra_keywords:
        keywords.extend(word.strip().lower() for word in extra_keywords.split(",") if word.strip())
    lookup = torch.zeros(vocab_size, dtype=torch.float32)
    for token_id in range(vocab_size):
        pieces = []
        try:
            piece = tokenizer.convert_ids_to_tokens(token_id)
            if piece is not None:
                pieces.append(str(piece))
        except Exception:
            pass
        try:
            decoded = tokenizer.decode([token_id], skip_special_tokens=True)
            if decoded:
                pieces.append(decoded)
        except Exception:
            pass
        text = " ".join(pieces).lower()
        if any(keyword in text for keyword in keywords):
            lookup[token_id] = 1.0
    return lookup


def save_eagle_head(accelerator, model, config, outdir):
    os.makedirs(outdir, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    save_file(unwrapped.state_dict(), os.path.join(outdir, "model.safetensors"))
    with open(os.path.join(outdir, "config.json"), "w", encoding="utf-8") as dst:
        json.dump(config.to_dict(), dst, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basepath", required=True)
    parser.add_argument("--pointllm-repo-path", default=None)
    parser.add_argument("--configpath", default=os.path.join(os.path.dirname(__file__), "pointllm_7B_config.json"))
    parser.add_argument("--tmpdir", required=True)
    parser.add_argument("--cpdir", required=True)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--num-warmup-steps", type=int, default=2000)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-freq", type=int, default=5)
    parser.add_argument("--p-w", type=float, default=1.0)
    parser.add_argument("--v-w", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--no-data-noise", action="store_true")
    parser.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--init-head", default=None)
    parser.add_argument("--acceptance-margin-weight", type=float, default=0.0)
    parser.add_argument("--acceptance-margin", type=float, default=1.0)
    parser.add_argument("--max-files", type=int, default=0, help="Use only the first N training files for quick MAT experiments.")
    parser.add_argument("--point-aware-weight", type=float, default=0.0, help="Enable point-aware token weighting when > 0.")
    parser.add_argument("--visual-token-weight", type=float, default=1.0)
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--uncertainty-margin", type=float, default=2.0)
    parser.add_argument("--visual-keywords", default=None)
    parser.add_argument("--rollout-steps", type=int, default=1, help="Train draft on its own multi-step hidden trajectory.")
    parser.add_argument("--rollout-weight", type=float, default=0.0)
    parser.add_argument("--rollout-decay", type=float, default=0.7)
    parser.add_argument("--point-rollout-adaptive", action="store_true", help="Weight rollout by point-scene difficulty, answer length, and teacher uncertainty.")
    parser.add_argument("--rollout-length-weight", type=float, default=0.25)
    parser.add_argument("--rollout-uncertainty-weight", type=float, default=0.35)
    parser.add_argument("--rollout-visual-weight", type=float, default=0.25)
    parser.add_argument("--rollout-coverage-weight", type=float, default=0.0, help="Increase rollout weight for samples with poor semantic-octree coverage.")
    parser.add_argument("--rollout-adaptive-min", type=float, default=0.7)
    parser.add_argument("--rollout-adaptive-max", type=float, default=1.8)
    args = parser.parse_args()

    add_pointllm_to_path(args.pointllm_repo_path)
    # Register PointLLM config with transformers if the source package is available.
    try:
        import pointllm.model  # noqa: F401
    except Exception:
        pass

    set_seed(0)
    accelerator = Accelerator(
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    datapath = list_files(args.tmpdir)
    if args.max_files and args.max_files > 0:
        datapath = datapath[: args.max_files]
    if len(datapath) < 2:
        raise ValueError(f"Need at least two training files under {args.tmpdir}, got {len(datapath)}")

    split = max(1, int(len(datapath) * 0.95))
    traindatapath = datapath[:split]
    testdatapath = datapath[split:] or datapath[: min(len(datapath), args.bs)]
    aug = None if args.no_data_noise else AddUniformNoise(std=args.noise_std)

    train_loader = DataLoader(
        CustomDataset(traindatapath, max_len=args.max_len, transform=aug),
        batch_size=args.bs,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        CustomDataset(testdatapath, max_len=args.max_len),
        batch_size=args.bs,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    baseconfig = AutoConfig.from_pretrained(args.basepath)
    config = EConfig.from_pretrained(args.configpath)
    config.architectures = ["PointLLMTreeForCausalLM"]
    config.hidden_size = getattr(baseconfig, "hidden_size", config.hidden_size)
    config.intermediate_size = getattr(baseconfig, "intermediate_size", config.intermediate_size)
    config.num_attention_heads = getattr(baseconfig, "num_attention_heads", config.num_attention_heads)
    config.num_key_value_heads = getattr(baseconfig, "num_key_value_heads", config.num_key_value_heads)
    config.vocab_size = getattr(baseconfig, "vocab_size", config.vocab_size)
    config.max_position_embeddings = getattr(
        baseconfig, "max_position_embeddings", config.max_position_embeddings
    )
    model = Model(config, load_emb=True, path=args.basepath)
    if args.init_head:
        init_path = args.init_head
        if os.path.isdir(init_path):
            init_path = os.path.join(init_path, "model.safetensors")
        incompatible = model.load_state_dict(load_file(init_path, device="cpu"), strict=False)
        print(
            f"[INFO] Initialized draft head from {init_path}; "
            f"missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )

    hidden_size = getattr(baseconfig, "hidden_size", config.hidden_size)
    vocab_size = getattr(baseconfig, "vocab_size", config.vocab_size)
    head = nn.Linear(hidden_size, vocab_size, bias=False)
    head.weight.data = load_weight_from_checkpoint(args.basepath, "lm_head.weight")
    head.eval()
    for param in head.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(args.basepath)
    visual_token_lookup = build_visual_token_lookup(tokenizer, vocab_size, args.visual_keywords)
    print(f"[INFO] point-aware visual token ids: {int(visual_token_lookup.sum().item())}/{vocab_size}")

    criterion = nn.SmoothL1Loss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    total_steps = max(1, args.num_epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.num_warmup_steps, total_steps),
        num_training_steps=total_steps,
    )

    model, head, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader, scheduler
    )
    visual_token_lookup = visual_token_lookup.to(accelerator.device)

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for data in tqdm(train_loader, disable=not accelerator.is_local_main_process):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                model_dtype = next(model.parameters()).dtype
                predict = model(
                    data["hidden_states"].to(model_dtype),
                    input_ids=data["input_ids"],
                    inputs_embeds=data["inputs_embeds"].to(model_dtype),
                    attention_mask=data["attention_mask"],
                )
                with torch.no_grad():
                    target = data["target"].to(model_dtype)
                    target_head = head(target).float()
                    target_p = nn.Softmax(dim=2)(target_head).detach()
                out_head = head(predict).float()
                out_logp = nn.LogSoftmax(dim=2)(out_head)
                teacher_top2 = torch.topk(target_head, k=2, dim=-1)
                token_weight = data["loss_mask"].float().to(out_head.device)
                if args.point_aware_weight > 0:
                    target_ids = data["input_ids"].to(out_head.device).clamp(min=0, max=vocab_size - 1)
                    visual_mask = visual_token_lookup[target_ids].float()
                    teacher_margin = teacher_top2.values[..., 0] - teacher_top2.values[..., 1]
                    uncertainty = torch.relu(args.uncertainty_margin - teacher_margin) / max(args.uncertainty_margin, 1e-5)
                    risk = args.visual_token_weight * visual_mask + args.uncertainty_weight * uncertainty
                    token_weight = token_weight * (1.0 + args.point_aware_weight * risk)
                loss_mask = token_weight[:, :, None]
                ploss_token = -torch.sum(target_p * out_logp, 2)
                rollout_factor = build_point_rollout_factor(
                    args, data, token_weight, visual_token_lookup, target_p
                )
                ploss = torch.sum(token_weight * ploss_token) / (token_weight.sum() + 1e-5)
                vloss = criterion(predict, target)
                vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (token_weight.sum() + 1e-5)
                teacher_token = teacher_top2.indices[..., 0]
                teacher_confidence = torch.sigmoid(
                    teacher_top2.values[..., 0] - teacher_top2.values[..., 1]
                )
                student_top2 = torch.topk(out_head, k=2, dim=-1)
                positive = out_head.gather(-1, teacher_token[..., None]).squeeze(-1)
                strongest_negative = torch.where(
                    student_top2.indices[..., 0] == teacher_token,
                    student_top2.values[..., 1],
                    student_top2.values[..., 0],
                )
                token_mask = token_weight
                acceptance_margin_loss = (
                    torch.relu(args.acceptance_margin - positive + strongest_negative)
                    * teacher_confidence
                    * token_mask
                ).sum() / (token_mask.sum() + 1e-5)
                rollout_loss = out_head.new_tensor(0.0)
                if args.rollout_weight > 0 and args.rollout_steps > 1:
                    rollout_hidden = predict
                    for rollout_index in range(2, args.rollout_steps + 1):
                        if rollout_hidden.shape[1] <= 1:
                            break
                        rollout_hidden = rollout_hidden[:, :-1, :]
                        rollout_input_ids = data["input_ids"][:, rollout_index - 1 :]
                        rollout_inputs_embeds = data["inputs_embeds"][:, rollout_index - 1 :, :].to(model_dtype)
                        rollout_attention = data["attention_mask"][:, rollout_index - 1 :]
                        rollout_target = data["target"][:, rollout_index - 1 :, :].to(model_dtype)
                        rollout_token_weight = token_weight[:, rollout_index - 1 :]
                        common_len = min(
                            rollout_hidden.shape[1],
                            rollout_input_ids.shape[1],
                            rollout_inputs_embeds.shape[1],
                            rollout_target.shape[1],
                            rollout_token_weight.shape[1],
                        )
                        if common_len <= 0:
                            break
                        rollout_hidden = rollout_hidden[:, :common_len, :]
                        rollout_input_ids = rollout_input_ids[:, :common_len]
                        rollout_inputs_embeds = rollout_inputs_embeds[:, :common_len, :]
                        rollout_attention = rollout_attention[:, :common_len]
                        rollout_target = rollout_target[:, :common_len, :]
                        rollout_token_weight = rollout_token_weight[:, :common_len]
                        rollout_pred = model(
                            rollout_hidden,
                            input_ids=rollout_input_ids,
                            inputs_embeds=rollout_inputs_embeds,
                            attention_mask=rollout_attention,
                        )
                        with torch.no_grad():
                            rollout_target_p = nn.Softmax(dim=2)(head(rollout_target).float()).detach()
                        rollout_logp = nn.LogSoftmax(dim=2)(head(rollout_pred).float())
                        rollout_ploss_token = -torch.sum(rollout_target_p * rollout_logp, 2)
                        sample_denom = rollout_token_weight.sum(dim=1).clamp_min(1e-5)
                        sample_loss = torch.sum(rollout_token_weight * rollout_ploss_token, dim=1) / sample_denom
                        if rollout_factor is not None:
                            sample_loss = sample_loss * rollout_factor[: sample_loss.shape[0]]
                        step_loss = sample_loss.mean()
                        rollout_loss = rollout_loss + (args.rollout_decay ** (rollout_index - 2)) * step_loss
                        rollout_hidden = rollout_pred
                loss = (
                    args.v_w * vloss
                    + args.p_w * ploss
                    + args.acceptance_margin_weight * acceptance_margin_loss
                    + args.rollout_weight * rollout_loss
                )
                accelerator.backward(loss)
                accelerator.clip_grad_value_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
            train_loss += loss.detach().float().item()
            train_batches += 1

        if accelerator.is_local_main_process:
            print(f"Epoch {epoch + 1}/{args.num_epochs} train_loss={train_loss / max(train_batches, 1):.4f}")

        if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.num_epochs:
            model.eval()
            eval_loss = 0.0
            eval_batches = 0
            eval_correct = 0.0
            eval_tokens = 0.0
            for data in tqdm(test_loader, disable=not accelerator.is_local_main_process):
                with torch.no_grad():
                    model_dtype = next(model.parameters()).dtype
                    predict = model(
                        data["hidden_states"].to(model_dtype),
                        input_ids=data["input_ids"],
                        inputs_embeds=data["inputs_embeds"].to(model_dtype),
                        attention_mask=data["attention_mask"],
                    )
                    target_head = head(data["target"].to(model_dtype)).float()
                    target_p = nn.Softmax(dim=2)(target_head).detach()
                    out_logp = nn.LogSoftmax(dim=2)(head(predict).float())
                    loss_mask = data["loss_mask"][:, :, None]
                    ploss = -torch.sum(torch.sum(loss_mask * target_p * out_logp, 2)) / (
                        loss_mask.sum() + 1e-5
                    )
                    token_mask = data["loss_mask"].bool()
                    eval_correct += (
                        (out_logp.argmax(dim=-1) == target_head.argmax(dim=-1)) & token_mask
                    ).sum().item()
                    eval_tokens += token_mask.sum().item()
                    eval_loss += ploss.detach().float().item()
                    eval_batches += 1
            if accelerator.is_local_main_process:
                print(
                    f"Epoch {epoch + 1}/{args.num_epochs} "
                    f"eval_ploss={eval_loss / max(eval_batches, 1):.4f} "
                    f"eval_top1={eval_correct / max(eval_tokens, 1):.4f}"
                )
                save_eagle_head(accelerator, model, config, args.cpdir)
            accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
