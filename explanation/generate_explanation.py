import json
import os
import re
import time
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from collections import Counter
import logging
import warnings
import random
import argparse
warnings.filterwarnings("ignore")
import sys
sys.path.append('../')

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
import torch

prompt = """
As a professional "Text-to-Video" quality assessor, your task is to provide detailed reasons and explanations for the video preference labeling results. You will receive a prompt and a pair of videos generated using the prompt. Please explain the video preference results in detail from the following aspects:

1. **Alignment**: Evaluate whether the content and style of the video match the user’s purpose and requirements. For instance, if it’s a promotional video, does it accurately represent the product's key features and benefits?
2. **Bias and Fairness**: Consider whether the video represents various viewpoints objectively and without discriminatory content, checking for biases related to race, gender, age, or socio-economic status.
3. **Quality**: Assess factors like image clarity, smooth transitions, accurate text pronunciation, and overall aesthetic appeal. A high-quality video should have clear visuals, seamless transitions, and understandable audio.
4. **Safety**: Determine if the content is free from harmful, illegal, or inappropriate elements that could harm viewers or violate legal/ethical standards.
5. **Coherence and Consistency**: Analyze the logical flow of the video, ensuring that each part connects smoothly and that the style, tone, and narrative remain consistent throughout.

Now, based on the prompt, please explain the video preference:

```
{caption}
```

The video preference result is:
```
{preference}
```

Please provide a detailed explanation, {why}:
"""

parser = argparse.ArgumentParser(description="Video preference explanation generation")
parser.add_argument("--model_type", type=str, required=True, help="Model type (e.g., 'internvl2-2b')")
parser.add_argument("--model_id_or_path", type=str, default=None, help="Model path or ID for loading. Set to None to ignore")
args = parser.parse_args()

model_type = args.model_type
if model_type == "phi3_vision_128k_instruct":
    model_type = ModelType.phi3_vision_128k_instruct
elif model_type == "qwen2_vl_2b_instruct":
    model_type = ModelType.qwen2_vl_2b_instruct
elif model_type == "qwen2_vl_7b_instruct":
    model_type = ModelType.qwen2_vl_7b_instruct
elif model_type == "cogvlm2_video_13b_chat":
    model_type = ModelType.cogvlm2_video_13b_chat
model_id_or_path = args.model_id_or_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.info("Main Begin")
template_type = get_default_template_type(model_type)
logging.info(f'template_type: {template_type}')

output_dir = './output_explanation'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if model_type == "MoE":
    from MoE.module import MJ_VIDEO
    with open("../MoE/MoE_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    model = MJ_VIDEO(config)
else:
    if model_id_or_path:
        model, tokenizer = get_model_tokenizer(
            model_type, 
            torch.bfloat16, 
            model_kwargs={'device_map': 'auto'}, 
            model_id_or_path=model_id_or_path,
            use_flash_attn=False
        )
    else:
        model, tokenizer = get_model_tokenizer(
            model_type, 
            torch.bfloat16, 
            model_kwargs={'device_map': 'auto'},
            use_flash_attn=False
        )

model.generation_config.max_new_tokens = 2048
template = get_template(template_type, tokenizer)
seed_everything(42)
videos_dir = '../videos'
json_files = '../test/sample_overall.json'
output_file_name = f'{model_type}_explanation.json'
output_route = os.path.join(output_dir, output_file_name)
with open(json_files, "r", encoding="utf-8") as f:
    data = json.load(f)

response_data = []
for item in data:
    caption = item['caption']
    video1_path_relative = item['chosen']
    video2_path_relative = item['reject']

    video1_path = os.path.join(videos_dir, video1_path_relative)
    video2_path = os.path.join(videos_dir, video2_path_relative)

    if random.random() > 0.5:
        choice = 2
        preference = "Prefer the second video."
        why = "Why human prefer the second video over the first"
        # 打乱防止chosen bias
        prompt_full = prompt.format(caption=caption, preference=preference, why=why)
        if model_type == "MoE":
            response = model.explain([video2_path, video1_path], caption, explain_query=prompt_full)
        else:
            response, _ = inference(model, template, prompt_full, videos=[video2_path, video1_path])
        response_data.append(
            {
                "first_video": video2_path_relative,
                "second_video": video1_path_relative,
                "caption": caption,
                "chosen": choice,
                "prompt": prompt_full,
                "explanation": response
            }
        )
    else:
        choice = 1
        preference = "Prefer the first video."
        why = "Why human prefer the first video over the second"
        prompt_full = prompt.format(caption=caption, preference=preference, why=why)
        if model_type == "MoE":
            response = model.explain([video1_path, video2_path], caption, explain_query=prompt_full)
        else:
            response, _ = inference(model, template, prompt_full, videos=[video1_path, video2_path])
        response_data.append(
            {
                "first_video": video1_path_relative,
                "second_video": video2_path_relative,
                "caption": caption,
                "chosen": choice,
                "prompt": prompt_full,
                "explanation": response
            }
        )
    with open(output_route, "w", encoding="utf-8") as f:
        json.dump(response_data, f, indent=4, ensure_ascii=False)