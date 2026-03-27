import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model import *
from pointllm.data import load_objaverse_point_cloud

import os

def load_assistant_model(assistant_model_name, torch_dtype):
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

def _softmax_logits(logits, temperature=1.0):
    if temperature <= 0:
        temperature = 1.0
    return torch.softmax(logits / temperature, dim=-1)

def _sample_from_probs(probs):
    return torch.multinomial(probs, num_samples=1)

def _entropy_from_probs(probs):
    probs = probs.clamp_min(1e-8)
    return -(probs * probs.log()).sum(dim=-1)

def _estimate_point_complexity(point_clouds):
    # heuristic geometry complexity score in [0, 1]
    if point_clouds is None:
        return 0.5
    xyz = point_clouds[..., :3]
    centered = xyz - xyz.mean(dim=1, keepdim=True)
    spread = centered.norm(dim=-1).mean().item()
    # normalize with a soft clipping
    return float(max(0.0, min(1.0, spread / (spread + 1.0))))

def _compute_adaptive_k(base_k, entropy, point_complexity, min_k=2, max_k=16):
    # high entropy / high complexity -> smaller drafting window
    entropy_scale = 1.0 - float(max(0.0, min(1.0, entropy)))
    complexity_scale = 1.0 - 0.5 * float(max(0.0, min(1.0, point_complexity)))
    k = int(round(base_k * entropy_scale * complexity_scale))
    return max(min_k, min(max_k, k))

def _acceptance_calibration(entropy, point_complexity, base=1.0):
    # uncertain region -> conservative acceptance
    entropy_penalty = 0.35 * float(max(0.0, min(1.0, entropy)))
    complexity_penalty = 0.20 * float(max(0.0, min(1.0, point_complexity)))
    return max(0.35, min(1.0, base - entropy_penalty - complexity_penalty))

def _assistant_propose_tokens(assistant_model, input_ids, k, do_sample, temperature, top_k, top_p):
    propose_tokens = []
    q_dists = []
    running_ids = input_ids

    for _ in range(k):
        outputs = assistant_model(input_ids=running_ids, use_cache=False, return_dict=True)
        logits = outputs.logits[:, -1, :]
        probs = _softmax_logits(logits, temperature=temperature)

        if top_k > 0 and top_k < probs.shape[-1]:
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
            if do_sample:
                sampled_idx = _sample_from_probs(topk_probs)
                next_token = torch.gather(topk_indices, -1, sampled_idx)
            else:
                next_token = topk_indices[:, :1]
        else:
            if do_sample:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                nucleus_mask = cum_probs <= top_p
                nucleus_mask[:, 0] = True
                filtered_probs = torch.where(nucleus_mask, sorted_probs, torch.zeros_like(sorted_probs))
                filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
                sampled_idx = _sample_from_probs(filtered_probs)
                next_token = torch.gather(sorted_idx, -1, sampled_idx)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

        propose_tokens.append(next_token)
        q_dists.append(probs)
        running_ids = torch.cat([running_ids, next_token], dim=1)

    draft_tokens = torch.cat(propose_tokens, dim=1)
    q_dists = torch.stack(q_dists, dim=1)
    return draft_tokens, q_dists

def speculative_decode_pointllm(
    model,
    assistant_model,
    input_ids,
    point_clouds,
    tokenizer,
    stop_str,
    max_new_tokens=256,
    num_assistant_tokens=8,
    min_assistant_tokens=2,
    max_assistant_tokens=16,
    adaptive_speculation=True,
    do_sample=True,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
):
    if assistant_model is None:
        return model.generate(
            input_ids,
            point_clouds=point_clouds,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    accepted_ids = input_ids
    prompt_len = input_ids.shape[1]
    point_complexity = _estimate_point_complexity(point_clouds)

    for _ in range(max_new_tokens):
        # target uncertainty at current step -> adaptive speculation policy
        step_outputs = model(
            input_ids=accepted_ids,
            point_clouds=point_clouds,
            use_cache=False,
            return_dict=True,
        )
        step_probs = _softmax_logits(step_outputs.logits[:, -1, :], temperature=temperature)
        # normalize by log(vocab_size) to map roughly into [0, 1]
        step_entropy = (_entropy_from_probs(step_probs) / torch.log(torch.tensor(step_probs.shape[-1], device=step_probs.device))).mean().item()

        cur_k = num_assistant_tokens
        if adaptive_speculation:
            cur_k = _compute_adaptive_k(
                base_k=num_assistant_tokens,
                entropy=step_entropy,
                point_complexity=point_complexity,
                min_k=min_assistant_tokens,
                max_k=max_assistant_tokens,
            )
        acceptance_calib = _acceptance_calibration(step_entropy, point_complexity)

        draft_tokens, q_dists = _assistant_propose_tokens(
            assistant_model=assistant_model,
            input_ids=accepted_ids,
            k=cur_k,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        combined_ids = torch.cat([accepted_ids, draft_tokens], dim=1)
        outputs = model(
            input_ids=combined_ids,
            point_clouds=point_clouds,
            use_cache=False,
            return_dict=True,
        )
        target_logits = outputs.logits[:, accepted_ids.shape[1]-1:combined_ids.shape[1]-1, :]
        p_probs = _softmax_logits(target_logits, temperature=temperature)

        rejected = False
        for j in range(draft_tokens.shape[1]):
            token_j = draft_tokens[:, j:j+1]
            p_j = torch.gather(p_probs[:, j, :], -1, token_j).squeeze(-1)
            q_j = torch.gather(q_dists[:, j, :], -1, token_j).squeeze(-1).clamp_min(1e-8)
            accept_ratio = torch.minimum(torch.ones_like(p_j), (p_j / q_j) * acceptance_calib)
            u = torch.rand_like(accept_ratio)

            if torch.all(u <= accept_ratio):
                accepted_ids = torch.cat([accepted_ids, token_j], dim=1)
                if accepted_ids.shape[1] - prompt_len >= max_new_tokens:
                    break
            else:
                diff = (p_probs[:, j, :] - q_dists[:, j, :]).clamp_min(0)
                diff_sum = diff.sum(dim=-1, keepdim=True)
                diff = torch.where(diff_sum > 0, diff / diff_sum, p_probs[:, j, :])
                sampled = _sample_from_probs(diff)
                accepted_ids = torch.cat([accepted_ids, sampled], dim=1)
                rejected = True
                break

        if accepted_ids.shape[1] - prompt_len >= max_new_tokens:
            break

        decoded = tokenizer.batch_decode(accepted_ids[:, prompt_len:], skip_special_tokens=True)[0]
        if stop_str and stop_str in decoded:
            break
        if accepted_ids[0, -1].item() == tokenizer.eos_token_id:
            break

        if rejected:
            continue

        outputs_next = model(
            input_ids=accepted_ids,
            point_clouds=point_clouds,
            use_cache=False,
            return_dict=True,
        )
        next_probs = _softmax_logits(outputs_next.logits[:, -1, :], temperature=temperature)
        next_token = _sample_from_probs(next_probs) if do_sample else torch.argmax(next_probs, dim=-1, keepdim=True)
        accepted_ids = torch.cat([accepted_ids, next_token], dim=1)

        decoded = tokenizer.batch_decode(accepted_ids[:, prompt_len:], skip_special_tokens=True)[0]
        if stop_str and stop_str in decoded:
            break
        if accepted_ids[0, -1].item() == tokenizer.eos_token_id:
            break

    return accepted_ids

def load_point_cloud(args):
    object_id = args.object_id
    print(f"[INFO] Loading point clouds using object_id: {object_id}")
    point_cloud = load_objaverse_point_cloud(args.data_path, object_id, pointnum=8192, use_color=True)
    
    return object_id, torch.from_numpy(point_cloud).unsqueeze_(0).to(torch.float32)

def init_model(args):
    # Model
    disable_torch_init()

    model_path = args.model_name 
    print(f'[INFO] Model name: {model_path}')

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = PointLLMLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=False, use_cache=True, torch_dtype=args.torch_dtype).cuda()
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    assistant_model = load_assistant_model(args.assistant_model_name, args.torch_dtype)

    model.eval()

    mm_use_point_start_end = getattr(model.config, "mm_use_point_start_end", False)
    # Add special tokens ind to model.point_config
    point_backbone_config = model.get_model().point_backbone_config
    
    if mm_use_point_start_end:
        if "v1" in model_path.lower():
            conv_mode = "vicuna_v1_1"
        else:
            raise NotImplementedError

        conv = conv_templates[conv_mode].copy()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    
    return model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv, assistant_model

def start_conversation(args, model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv, assistant_model=None):
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']
    # The while loop will keep running until the user decides to quit
    print("[INFO] Starting conversation... Enter 'q' to exit the program and enter 'exit' to exit the current conversation.")
    while True:
        print("-" * 80)
        # Prompt for object_id
        object_id = input("[INFO] Please enter the object_id or 'q' to quit: ")
        
        # Check if the user wants to quit
        if object_id.lower() == 'q':
            print("[INFO] Quitting...")
            break
        else:
            # print info
            print(f"[INFO] Chatting with object_id: {object_id}.")
        
        # Update args with new object_id
        args.object_id = object_id.strip()
        
        # Load the point cloud data
        try:
            id, point_clouds = load_point_cloud(args)
        except Exception as e:
            print(f"[ERROR] {e}")
            continue
        point_clouds = point_clouds.cuda().to(args.torch_dtype)

        # Reset the conversation template
        conv.reset()

        print("-" * 80)

        # Start a loop for multiple rounds of dialogue
        for i in range(100):
            # This if-else block ensures the initial question from the user is included in the conversation
            qs = input(conv.roles[0] + ': ')
            if qs == 'exit':
                break
            
            if i == 0:
                if mm_use_point_start_end:
                    qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + qs
                else:
                    qs = default_point_patch_token * point_token_len + '\n' + qs

            # Append the new message to the conversation history
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt])

            input_ids = torch.as_tensor(inputs.input_ids).cuda()

            stop_str = keywords[0]

            with torch.inference_mode():
                output_ids = speculative_decode_pointllm(
                    model=model,
                    assistant_model=assistant_model,
                    input_ids=input_ids,
                    point_clouds=point_clouds,
                    tokenizer=tokenizer,
                    stop_str=stop_str,
                    max_new_tokens=args.max_new_tokens,
                    num_assistant_tokens=args.num_assistant_tokens,
                    min_assistant_tokens=args.min_assistant_tokens,
                    max_assistant_tokens=args.max_assistant_tokens,
                    adaptive_speculation=not args.disable_adaptive_speculation,
                    do_sample=True,
                    temperature=1.0,
                    top_k=50,
                    top_p=0.95,
                )

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            # Append the model's response to the conversation history
            conv.pop_last_none_message()
            conv.append_message(conv.roles[1], outputs)
            print(f'{conv.roles[1]}: {outputs}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, \
       default="RunsenXu/PointLLM_7B_v1.2")

    parser.add_argument("--data_path", type=str, default="data/objaverse_data")
    parser.add_argument("--torch_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--assistant_model_name", type=str, default=None, help="Draft model name/path for speculative decoding.")
    parser.add_argument("--num_assistant_tokens", type=int, default=8, help="How many draft tokens to propose per step in speculative decoding.")
    parser.add_argument("--min_assistant_tokens", type=int, default=2, help="Minimum draft tokens per step when adaptive speculation is enabled.")
    parser.add_argument("--max_assistant_tokens", type=int, default=16, help="Maximum draft tokens per step when adaptive speculation is enabled.")
    parser.add_argument("--disable_adaptive_speculation", action="store_true", help="Disable uncertainty-aware adaptive speculation and use fixed num_assistant_tokens.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of generated tokens per answer.")

    args = parser.parse_args()

    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    args.torch_dtype = dtype_mapping[args.torch_dtype]

    model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv, assistant_model = init_model(args)
    
    start_conversation(args, model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv, assistant_model=assistant_model)
