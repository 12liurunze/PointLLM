from eagle_eye.model.ee_model import EeModel

import torch
from PIL import Image
import os
import time

# base_model_path = "/home/dhz/llava-v1.5-7b-hf"
# ee_model_path = "/home/dhz/tmp_model/EAGLE-EYE-LLaVA-7B-10k"

base_model_path = "/home/dhz/Qwen2.5-VL-7B-Instruct"
ee_model_path = "/home/dhz/tmp_model/EAGLE-EYE-Qwen2.5vl-7B-10k-video"
from qwen_vl_utils import process_vision_info

model = EeModel.from_pretrained(
    base_model_path=base_model_path,
    ee_model_path=ee_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    attn_implementation="eager"
)
model.eval()

url = "/home/dhz/K400/val_256/abseiling/_aQSjArgAqA.mkv"

con = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video":url,
                    "fps": 1.0,
                },
                {"type": "text", "text": "Describe what happen in the video?"},
            ],
        }
    ]
   
text = model.processor.apply_chat_template(con, tokenize=False, add_generation_prompt=True)

image_inputs, video_inputs, video_kwargs = process_vision_info(con, return_video_kwargs=True)
inputs = model.processor(text=text, videos=video_inputs,return_tensors="pt",**video_kwargs).to(model.base_model.device, torch.float16)

output_ids = model.eagenerate(**inputs,temperature=0.0, max_new_tokens=200)

output = model.processor.tokenizer.decode(output_ids[0][inputs.input_ids.shape[-1]:],skip_special_tokens=True)
print(output)
