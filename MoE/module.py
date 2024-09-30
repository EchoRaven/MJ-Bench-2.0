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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    transform = Compose([
        Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    return np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])


def load_video(video_path, bound=None, input_size=448, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=20)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    frame_indices = get_index(bound, fps, max_frame, num_segments=num_segments)
    transform = build_transform(input_size)

    pixel_values_list = []
    num_patches_list = []
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = img.resize((input_size, input_size), Image.BICUBIC)
        pixel_values = transform(img).unsqueeze(0)
        pixel_values_list.append(pixel_values)
        num_patches_list.append(pixel_values.shape[0])

    return torch.cat(pixel_values_list), num_patches_list


def process_video(video_path, prompt, model, tokenizer):
    pixel_values, num_patches_list = load_video(video_path, num_segments=8)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    video_prefix = ''.join([f'Frame{i + 1}: <image>\n' for i in range(len(num_patches_list))])
    question = video_prefix + prompt
    response, history = model.chat(
        tokenizer, pixel_values, question,
        generation_config={'max_new_tokens': 1024, 'do_sample': True},
        num_patches_list=num_patches_list, history=None, return_history=True
    )
    return response


def load_multiple_videos(video_paths, num_segments=8):
    pixel_values_list = []
    num_patches_lists = []

    for video_path in video_paths:
        # 加载每个视频的帧数据和patches列表
        pixel_values, num_patches_list = load_video(video_path, num_segments)
        pixel_values_list.append(pixel_values.to(torch.float32).cuda())
        num_patches_lists.append(num_patches_list)

    return pixel_values_list, num_patches_lists


def generate(dtype, model, video_paths, template, prompt, tokenizer):
    # 加载多个视频
    pixel_values_list, num_patches_lists = load_multiple_videos(video_paths, num_segments=8)

    # 为每个视频创建问题前缀
    video_prefixes = ""
    combined_patch_list = []
    for video_idx, num_patches_list in enumerate(num_patches_lists):
        video_prefix = f'Video-{video_idx + 1}: '.join(
            [f'Frame{i + 1}: <image>\n' for i in range(len(num_patches_list))])
        video_prefixes += video_prefix
        combined_patch_list.extend(num_patches_list)

    # 合并所有视频的前缀到最终的question中
    question = video_prefixes + template + prompt

    # 将多个视频的pixel_values合并为一个
    combined_pixel_values = torch.cat(pixel_values_list, dim=0)

    # 调用模型进行对话
    response, history = model.chat(
        tokenizer, combined_pixel_values, question,
        generation_config={'max_new_tokens': 1024, 'do_sample': True},
        num_patches_list=combined_patch_list,
        history=None, return_history=True
    )
    return response


class MJ_VIDEO:
    def __init__(self, config):
        self.config = config
        self.dtype = torch.bfloat16 if config["dtype"] == "bfloat16" else torch.float32
        self.base_model = AutoModel.from_pretrained(
            config["model_type"],
            cache_dir=config["model_path"],
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True
        )
        # define tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True, use_fast=False)

        # load prompt list
        with open(config["prompt_list"], "r", encoding="utf-8") as f:
            self.prompt_list = json.load(f)

        # define MoE
        self.inference_type = config["inference_type"]
        if config["inference_type"] == "high_speed":
            # define router
            self.router = PeftModel.from_pretrained(self.base_model, config["router_path"]).eval()

            # define experts
            self.expert_group = {}
            for key in config["experts"].keys():
                self.expert_group[key] = PeftModel.from_pretrained(self.base_model, config["experts"][key]).eval()

    def router_choice(self, video_paths, prompt):
        template = self.prompt_list["router"]
        if self.config["inference_type"] == "low_cost":
            self.router = PeftModel.from_pretrained(self.base_model, config["router_path"]).eval()
        response = generate(self.dtype, self.router, video_paths, template, prompt, self.tokenizer)


if __name__ == "__main__":
    with open("MoE_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    moe = MJ_VIDEO(config)
    video_paths = ["/remote_shome/snl/feilong/xiapeng/haibo/videos//safesora/8cd608c47b821009baf7cc43df12b183d6da0c8c9e7125717811fa00ad4930fa/4a4c1990b549e1221e0d663a21f2970b2628059161c82af1deb6d309cf0c9ea6.mp4", "/remote_shome/snl/feilong/xiapeng/haibo/videos//safesora/8cd608c47b821009baf7cc43df12b183d6da0c8c9e7125717811fa00ad4930fa/351b13217fc3ac1689b3f8b17356769ab7b9d36981db92462186a784f3bc57b2.mp4"]
    prompt = "2000 Documentary film in color showing dark hallway in house and kid in its center gets ripped apart from outside showing bloody monster"
    response = moe.router_choice(video_paths, prompt)
