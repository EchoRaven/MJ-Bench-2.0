import torch
import json
from swift.llm import (
    get_model_tokenizer, get_template, inference,
    get_default_template_type
)
from swift.tuners import Swift
from swift.utils import seed_everything
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import re
import json

expert, tokenizer =  get_model_tokenizer("internvl2-2b", torch.bfloat16,
                        model_kwargs={'device_map': 'auto'}, model_id_or_path="../videoRM/Internvl/pretrain/InternVL2-2B")
expert = Swift.from_pretrained(
        expert, "../videoRM/finetune_result/alignment_expert_Internvl2_2B_base_model_lora_short_analysis/internvl2-2b/v0-20241030-205249/checkpoint-1476", "alignmnet", inference_mode=True)

expert.generation_config.max_new_tokens = 1024

template_type = get_default_template_type("internvl2-2b")
template = get_template(template_type, tokenizer)
video_path_0 = ["../videos/safesora/59530a9c4f215a7a40e437d539e85c0d3085fa9700be7eb79e39e8e135a628a8/d813cf364340f0de682f775e60f83aacd997d4bcd393877284d54cde9f3b95df.mp4"]
video_path_1 = ["../videos/hpdbv2_videos/HPDv2_train_image1_6021.mp4"]

prompt0 = "Assess the video for Object, Attribute, Action, Count, and Location as good or bad.\nPrompt:\n\"Generate a 4K resolution video at 29 frames per second and an aspect ratio of 16:9. The video should depict a scene involving two men engaged in a conversation about treasure. The scene composition and other elements can be chosen autonomously; however, ensure the main focus remains on the dialogue between the two men and the topic of their discussion i.e., the treasure.\""

prompt1 = "Assess the video for Object, Attribute, Action, Count, and Location as good or bad.\nPrompt:\n\"Mario from super Mario brothers but he sitting in the real world at a desk arguing about a contract with a real person who is not a character from a video game, they argue over details of the contract on the desk in a realistic office, a cinematic scene from a movie\""

result, _ = inference(expert, template, prompt0, videos=video_path_0)
print(result)

result, _ = inference(expert, template, prompt1, videos=video_path_1)
print(result)