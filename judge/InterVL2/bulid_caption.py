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
from torchvision.transforms import Compose, Lambda, Resize, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

PROMPT_TEMPLATE = """
You are a professional video content descriptor.\n
Please briefly describe the video content:
"""

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
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
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
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    question = video_prefix + prompt
    response, history = model.chat(
        tokenizer, pixel_values, question,
        generation_config={'max_new_tokens': 1024, 'do_sample': True},
        num_patches_list=num_patches_list, history=None, return_history=True
    )
    return response


def main(args):
    model = AutoModel.from_pretrained(
        args.model_path,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)

    try:
        model.cuda()
    except Exception as e:
        print(f"Error in distributed setup: {str(e)}")
        exit(1)
    start_time = time.time()
    try:
        with open(args.json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {args.json_file_path} was not found.")
        exit(1)
    length = len(data)
    data = data[int(args.split * length / 10) : int((args.split + 1) * length / 10)]
    results = []
    for item in data:
        id = item['id']
        video_path_relative = item["video_path"]
        video_path = os.path.join(args.videos_dir, video_path_relative)
        result = process_video(video_path=video_path, prompt=PROMPT_TEMPLATE, model=model, tokenizer=tokenizer)
        results.append(
            {
                "id": id,
                "caption": result
            }
        )
        with open(args.results_file, 'w') as outfile:
            json.dump(results, outfile, indent=4, ensure_ascii=False)

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    with open(args.results_file, 'w') as outfile:
        json.dump(results, outfile, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate text-to-video models based on multiple criteria.")
    parser.add_argument('--json_file_path', type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument('--videos_dir', type=str, required=True, help="Directory where the videos are stored.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument('--cache_dir', type=str, required=True, help="Cache directory for the pre-trained model.")
    parser.add_argument('--results_file', type=str, default='result.json', help="File to save alignment results.")
    parser.add_argument('--split', type=int, required=True, help="split index.")

    args = parser.parse_args()
    main(args)