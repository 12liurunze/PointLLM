import torch
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration

from PIL import Image

from qwen_vl_utils import process_vision_info

model_dir = "/home/dhz/Qwen2.5-VL-7B-Instruct" 

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)

# url ="/home/dhz/eagle-eye/EAGLE_EYE/eagle_eye/example.jpg"
processor = AutoProcessor.from_pretrained(model_dir)





# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": url,
#             },
#             {"type": "text", "text": "Describe the image in detail."},
#         ],
#     }
# ]
# text = processor.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# ) 
# image = Image.open(url)
# inputs = processor(images=image, text=text , return_tensors='pt').to(model.base_model.device)


# # Inference: Generation of the output
# generated_ids = model.generate(**inputs, max_new_tokens=128)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text[0])

con = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video":"/home/dhz/tiny-Kinetics-400/train_256/abseiling/_4YTwq0-73Y_000044_000054.mp4",
                    "fps": 1.0,
                },
                {"type": "text", "text": "Describe what happen in the video?"},
            ],
        }
    ]
   
text = processor.apply_chat_template(con, tokenize=False, add_generation_prompt=True)

image_inputs, video_inputs, video_kwargs = process_vision_info(con, return_video_kwargs=True)
inputs = processor(text=text, videos=video_inputs,return_tensors="pt",**video_kwargs).to(model.device, torch.float16)


generated_ids = model.generate(**inputs, max_new_tokens=128)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)