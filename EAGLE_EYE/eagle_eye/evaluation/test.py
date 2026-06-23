import argparse
import json
import os
import time
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
import time
import shortuuid
from tqdm import tqdm
from PIL import Image
import pandas as pd
from transformers import AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map
from eagle_eye.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
import torch
#try:
from eagle_eye.model.utils import *
from eagle_eye.model.ee_model import EeModel
from eagle_eye.model.kv_cache import initialize_past_key_values
from eagle_eye.model.choices import *
import numpy as np
from qwen_vl_utils import process_vision_info
print(torch.distributed.is_available())
def ee_forward(input_ids, pixel_values_videos, video_grid_thw, model, tokenizer, tree_choices, logits_processor=None, max_steps=512):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()
    model.ee_layer.reset_kv()
    print(pixel_values_videos.shape)
    if hasattr(model, "tree_choices") and model.tree_choices == tree_choices:
        tree_buffers = model.tree_buffers
    else:
        tree_buffers = generate_tree_buffers(
            tree_choices,
            device=model.language_model.layers[-1].self_attn.q_proj.weight.device,
        )
        tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
            model.lm_head.weight.device
        )
    model.tree_buffers = tree_buffers
    model.tree_choices = tree_choices

    # Initialize the past key and value states
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        # Reset the past key and value states
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.language_model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    reset_tree_mode(model.language_model)

    tree_logits, logits, hidden_state, sample_token = initialize_tree(
        input_ids=input_ids,
        pixel_values_videos=pixel_values_videos,
        model=model,
        tree_attn_mask=tree_buffers["tree_attn_mask"],
        past_key_values=past_key_values,
        logits_processor=logits_processor,
        video_grid_thw=video_grid_thw,
    )
    new_token = 0

    for idx in range(max_steps):
        candidates, cart_candidates_prob, tree_candidates = generate_candidates(
            tree_logits=tree_logits,
            tree_indices=tree_buffers["tree_indices"],
            retrieve_indices=tree_buffers["retrieve_indices"],
            sample_token=sample_token,
            logits_processor=logits_processor,
        )
        logits, hidden_state_new, outputs = tree_decoding(
            model=model,
            tree_candidates=tree_candidates,
            past_key_values=past_key_values,
            tree_position_ids=tree_buffers["tree_position_ids"],
            input_ids=input_ids,
            retrieve_indices=tree_buffers["retrieve_indices_head"],
        )
        best_candidate, accept_length, sample_p = evaluate_posterior(
            logits=logits,
            candidates=candidates,
            logits_processor=logits_processor,
            cart_candidates_prob=cart_candidates_prob,
            op=tree_logits[2],
            p_indices=tree_buffers["p_indices"],
            tree_candidates=tree_candidates,
            b_indices=tree_buffers["b_indices"],
        )
        input_ids,tree_logits,new_token,hidden_state,sample_token = update_inference_inputs(
            input_ids=input_ids,
            candidates=candidates,
            best_candidate=best_candidate,
            accept_length=accept_length,
            retrieve_indices=tree_buffers["retrieve_indices"],
            logits_processor=logits_processor,
            logits=logits,
            tree_logits=tree_logits,
            new_token=new_token,
            past_key_values_data_list=past_key_values_data,
            current_length_data=current_length_data,
            model=model,
            hidden_state=hidden_state,
            hidden_state_new=hidden_state_new,
            sample_p=sample_p,
        )
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > 1024:
            break
        if input_ids.shape[1] > 4096:
            break
    return input_ids, new_token, idx


def run_eval(
        base_model_path,
        ee_model_path,
        model_id,
        answer_file,
        max_new_token,
        max_gpu_memory,
        temperature,
        tree_choices,
):
   
    random.seed(42)  
    with open("/root/autodl-tmp/eagle-eye/EAGLE_EYE/eagle_eye/metadata.jsonl", "r") as f:
        metadata = [json.loads(line) for line in f]
    
    
    get_answers_func = get_model_answers
    ans_handles = []
    ans_handles.append(
        get_answers_func(
            base_model_path,
            ee_model_path,
            model_id,
            answer_file,
            max_new_token,
            max_gpu_memory,
            temperature,
            tree_choices,
            metadata
        )
    )

@torch.inference_mode()
def get_model_answers(
        base_model_path,
        ee_model_path,
        model_id,
        #questions,
        answer_file,
        #datapath,
        max_new_token,
        max_gpu_memory,
        temperature,
        tree_choices,
        chunk,
):
    
    config = AutoConfig.from_pretrained("/root/autodl-tmp/qwen2.5vl")

    # 空模型
    with init_empty_weights():
        empty_model = Qwen2_5_VLForConditionalGeneration._from_config(config)

    # 自动 device_map
    device_map = infer_auto_device_map(
        empty_model,
        max_memory={0: "24GiB", 1: "24GiB"},
        no_split_module_classes=["Qwen2_5_VLVisionBlock"],  # 保证单个block不被拆开
    )

    # 手动调整 visual 层的切分
    num_layers = len(empty_model.visual.blocks)  # 32
    for i in range(num_layers):
        target_gpu = 0 if i < num_layers // 2 else 1
        device_map[f"visual.blocks.{i}"] = target_gpu

    # merger 放到最后一个 GPU
    device_map["visual.merger"] = 1

    print(device_map)
    model = EeModel.from_pretrained(
        base_model_path=base_model_path,
        ee_model_path=ee_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        attn_implementation="eager",
        tp_plan=None
    )
    #print(len(model.visual_encoder.layers))
    tokenizer = model.get_tokenizer()

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    print(f"[Worker] Processing {len(chunk)} items...")
    print('warmup ...')
    
    if len(chunk) > 0:
        warmup_item = chunk[0]
        url = warmup_item['path']
        con = [
            {"role": "user", "content": [
                {"type": "video", "video": url, "max_pixels":448*448,"fps": 1.0},
                {"type": "text", "text": "Describe what happen in the video?"},
            ]}
        ]
        text = model.processor.apply_chat_template(con, tokenize=False, add_generation_prompt=True)
        _, video_inputs, video_kwargs = process_vision_info(con, return_video_kwargs=True)
        inputs = model.processor(text=text, videos=video_inputs, return_tensors="pt", **video_kwargs)
        print(inputs.input_ids.shape, inputs.pixel_values_videos.shape, inputs.video_grid_thw.shape)
        ee_forward(
            input_ids=inputs.input_ids,
            pixel_values_videos=inputs.pixel_values_videos,
            video_grid_thw=inputs.video_grid_thw,
            model=model,
            tokenizer=tokenizer,
            tree_choices=tree_choices,
            logits_processor=logits_processor,
        )
        print("[Worker] Warmup done.")
    torch.manual_seed(123)
    for item in chunk:
        url = item['path']
        print(url)
        con = [
            {"role": "user", "content": [
                {"type": "video", "video": url, "max_pixels":448*448,"fps": 1.0},
                {"type": "text", "text": "Describe what happen in the video?"},
            ]}
        ]
        # if item['duration'] > 1000:
        #     continue
        text = model.processor.apply_chat_template(con, tokenize=False, add_generation_prompt=True)
        _, video_inputs, video_kwargs = process_vision_info(con, return_video_kwargs=True)
        inputs = model.processor(text=text, videos=video_inputs, return_tensors="pt", **video_kwargs)
        
        torch.cuda.synchronize()
        start_time = time.time()
        output_ids, _, idx = ee_forward(
            input_ids=inputs.input_ids,
            pixel_values_videos=inputs.pixel_values_videos,
            video_grid_thw=inputs.video_grid_thw,
            model=model,
            tokenizer=tokenizer,
            tree_choices=tree_choices,
            logits_processor=logits_processor,
        )
        torch.cuda.synchronize()
        total_time = time.time() - start_time

        # decode
        output_ids = output_ids[0][len(inputs.input_ids[0]):]
        new_token=output_ids.shape[-1]
        output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
        if tokenizer.eos_token and output.find(tokenizer.eos_token) > 0:
            output = output[: output.find(tokenizer.eos_token)]
        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()
   
            # Dump answers
        
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "response":output,
                "idx": idx,
                "new_tokens": new_token,
                "wall_time": total_time,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ee-model-path",
        type=str,
        default="/root/autodl-tmp/qwen",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--base-model-path", type=str, default="/root/autodl-tmp/qwen2.5vl",
                        help="1")
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument("--model-id", type=str, default="Qwen2.5-VL-7B-Instruct-video-fp16-ee")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="COCO-caption",
        help="The name of the benchmark question set.",
    )
    
    parser.add_argument("--answer-dir", type=str,default="/root/autodl-tmp/eagle-eye/EAGLE_EYE/eagle_eye/outputs",help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )

    args = parser.parse_args()

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    args.tree_choices = eval(args.tree_choices)
    answer_file = os.path.join(args.answer_dir, f"{args.model_id}.jsonl")
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    print(f"Output to {answer_file}")
    run_eval(
        base_model_path=args.base_model_path,
        ee_model_path=args.ee_model_path,
        model_id=args.model_id,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        max_gpu_memory=args.max_gpu_memory,
        temperature=args.temperature,
        tree_choices=args.tree_choices,
    )