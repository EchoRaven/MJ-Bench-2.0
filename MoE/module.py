import json
import os
import re
import time
import argparse
import torch
from accelerate import PartialState
import numpy as np
from decord import VideoReader, cpu
from PIL import Image
import torch.nn as nn
from torchvision.transforms import Compose, Lambda, Resize, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel, PeftConfig
import os
from swift.llm import (
    get_model_tokenizer, get_template, inference,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything


class MJ_VIDEO:
    def __init__(self, config):
        self.config = config
        self.dtype = torch.bfloat16 if config["dtype"] == "bfloat16" else torch.float32
        template_type = get_default_template_type(config["model_type"])
        self.template = get_template(template_type, self.tokenizer)
        seed_everything(42)

        # load prompt list
        with open(config["prompt_list"], "r", encoding="utf-8") as f:
            self.prompt_list = json.load(f)

        # define MoE
        self.inference_type = config["inference_type"]
        if config["inference_type"] == "high_speed":
            # define router
            self.router, _ = get_model_tokenizer(config["router_path"], self.dtype,
                                       model_kwargs={'device_map': 'auto'})
            self.router.generation_config.max_new_tokens = 1024
            # define experts
            self.expert_group = {}
            for key in config["experts"].keys():
                self.expert_group[key], _ = get_model_tokenizer(config["experts"][key], self.dtype,
                                       model_kwargs={'device_map': 'auto'})
                self.self.expert_group[key].generation_config.max_new_tokens = 1024

    def router_choice(self, video_paths, prompt):
        router_template = self.prompt_list["router"]
        if self.config["inference_type"] == "low_cost":
            self.router, _ = get_model_tokenizer(config["router_path"], self.dtype,
                                       model_kwargs={'device_map': 'auto'})
            self.router.generation_config.max_new_tokens = 1024
        response, _ = inference(self.router, self.template, router_template + prompt, videos=video_paths)  # chat with image
        return response


if __name__ == "__main__":
    with open("MoE_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    moe = MJ_VIDEO(config)
    video_paths = ["/remote_shome/snl/feilong/xiapeng/haibo/videos//safesora/8cd608c47b821009baf7cc43df12b183d6da0c8c9e7125717811fa00ad4930fa/4a4c1990b549e1221e0d663a21f2970b2628059161c82af1deb6d309cf0c9ea6.mp4", "/remote_shome/snl/feilong/xiapeng/haibo/videos//safesora/8cd608c47b821009baf7cc43df12b183d6da0c8c9e7125717811fa00ad4930fa/351b13217fc3ac1689b3f8b17356769ab7b9d36981db92462186a784f3bc57b2.mp4"]
    prompt = "2000 Documentary film in color showing dark hallway in house and kid in its center gets ripped apart from outside showing bloody monster"
    response = moe.router_choice(video_paths, prompt)
