import argparse
import json
import os
import sys
from typing import Any, Dict, List

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, get_linear_schedule_with_warmup

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
            (inputs_embeds[:, 1:, :], torch.zeros(1, 1, inputs_embeds.shape[2])),
            dim=1,
        )
        target = torch.cat((hidden_state[:, 1:, :], torch.zeros(1, 1, hidden_state.shape[2])), dim=1)

        new_data = {
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "target": target,
            "hidden_state_big": hidden_state,
            "input_ids": input_ids_target,
            "inputs_embeds": inputs_embeds_target,
        }
        if self.transform:
            new_data = self.transform(new_data)
        return new_data


class DataCollatorWithPadding:
    @staticmethod
    def paddingtensor(intensors, length):
        _, n, dim = intensors.shape
        return torch.cat((intensors, torch.zeros(1, length - n, dim)), dim=1)

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
        }


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
    args = parser.parse_args()

    add_pointllm_to_path(args.pointllm_repo_path)
    # Register PointLLM config with transformers if the source package is available.
    try:
        import pointllm.model  # noqa: F401
    except Exception:
        pass

    set_seed(0)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )
    if accelerator.is_main_process:
        print(
            f"[train_pointllm] distributed world_size={accelerator.num_processes} "
            f"per_device_batch_size={args.bs} mixed_precision={args.mixed_precision}"
        )

    datapath = list_files(args.tmpdir)
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

    hidden_size = getattr(baseconfig, "hidden_size", config.hidden_size)
    vocab_size = getattr(baseconfig, "vocab_size", config.vocab_size)
    head = nn.Linear(hidden_size, vocab_size, bias=False)
    head.weight.data = load_weight_from_checkpoint(args.basepath, "lm_head.weight")
    head.eval()
    for param in head.parameters():
        param.requires_grad = False

    criterion = nn.SmoothL1Loss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    model, head, optimizer, train_loader, test_loader = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader
    )
    # Accelerate shards each loader across ranks. Build the scheduler after
    # preparation so its step count matches the actual per-rank iterations.
    total_steps = max(1, args.num_epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.num_warmup_steps, total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for data in tqdm(train_loader, disable=not accelerator.is_local_main_process):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                predict = model(
                    data["hidden_states"],
                    input_ids=data["input_ids"],
                    inputs_embeds=data["inputs_embeds"],
                    attention_mask=data["attention_mask"],
                )
                with torch.no_grad():
                    target_head = head(data["target"]).float()
                    target_p = nn.Softmax(dim=2)(target_head).detach()
                out_head = head(predict).float()
                out_logp = nn.LogSoftmax(dim=2)(out_head)
                loss_mask = data["loss_mask"][:, :, None]
                ploss = -torch.sum(torch.sum(loss_mask * target_p * out_logp, 2)) / (
                    loss_mask.sum() + 1e-5
                )
                vloss = criterion(predict, data["target"])
                vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum() + 1e-5)
                loss = args.v_w * vloss + args.p_w * ploss
                accelerator.backward(loss)
                accelerator.clip_grad_value_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
            train_loss += loss.detach().float().item()
            train_batches += 1

        train_stats = torch.tensor(
            [train_loss, float(train_batches)],
            device=accelerator.device,
            dtype=torch.float64,
        )
        train_stats = accelerator.reduce(train_stats, reduction="sum")
        if accelerator.is_main_process:
            print(
                f"Epoch {epoch + 1}/{args.num_epochs} "
                f"train_loss={train_stats[0].item() / max(train_stats[1].item(), 1.0):.4f}"
            )

        if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.num_epochs:
            model.eval()
            eval_loss_numerator = torch.zeros((), device=accelerator.device)
            eval_token_count = torch.zeros((), device=accelerator.device)
            for data in tqdm(test_loader, disable=not accelerator.is_local_main_process):
                with torch.no_grad():
                    predict = model(
                        data["hidden_states"],
                        input_ids=data["input_ids"],
                        inputs_embeds=data["inputs_embeds"],
                        attention_mask=data["attention_mask"],
                    )
                    target_head = head(data["target"]).float()
                    target_p = nn.Softmax(dim=2)(target_head).detach()
                    out_logp = nn.LogSoftmax(dim=2)(head(predict).float())
                    loss_mask = data["loss_mask"][:, :, None]
                    eval_loss_numerator += -torch.sum(
                        torch.sum(loss_mask * target_p * out_logp, 2)
                    )
                    eval_token_count += loss_mask.sum()
            eval_stats = torch.stack(
                (eval_loss_numerator.double(), eval_token_count.double())
            )
            eval_stats = accelerator.reduce(eval_stats, reduction="sum")
            if accelerator.is_main_process:
                print(
                    f"Epoch {epoch + 1}/{args.num_epochs} "
                    f"eval_ploss={eval_stats[0].item() / max(eval_stats[1].item(), 1.0):.4f}"
                )
                save_eagle_head(accelerator, model, config, args.cpdir)
            accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
