import json
import os
import re
import time
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from collections import Counter
import sys
import logging
sys.path.append('../')


from swift.llm import (
    get_model_tokenizer, get_template, inference,ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
import torch

if not os.path.exists('./output_double'):
    os.mkdir('./output_double')

from MoE.module import MJ_VIDEO, convert_to_json_format
with open("../MoE/MoE_config_2B_2.json", "r", encoding="utf-8") as f:
    config = json.load(f)
model = MJ_VIDEO(config)
seed_everything(42)

json_file_path = "../test/overall.json"
videos_dir = '../videos'
with open(json_file_path, 'r') as f:
    data = json.load(f)

result_file = "router_judge_result.json"
judge_data = []

for item in data:
        try:
            caption = item['caption']
            video0_path_relative = item['chosen']
            video1_path_relative = item['reject']
            video0_path = os.path.join(videos_dir, video0_path_relative)
            video1_path = os.path.join(videos_dir, video1_path_relative)
            better = item['better']
            chosen = convert_to_json_format(model.router_choice([video0_path, video1_path], caption, "pair_prompt_template"))
            judge_data.append(
                    {
                        "video0": video0_path_relative,
                        "video1": video1_path_relative,
                        "caption": caption,
                        "better": better,
                        "chosen": chosen
                    }
            )
            with open(result_file, "w", encoding="utf-8") as f:
                 json.dump(judge_data, f, indent=4, ensure_ascii=False)
        except:
            continue