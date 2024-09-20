import os
import json
import argparse
from accelerate import PartialState
from diffusers import DiffusionPipeline
from PIL import Image
import torch
import logging
from tqdm import tqdm
import cv2
import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def process_video(entry, pipeline, video_path, frame_count, frame_duration, format):
    """Process each entry to generate video and save it."""
    if "image_path" in entry:
        image = Image.open(entry["image_path"]).convert("RGB")
    elif "image" in entry:
        image = entry["image"].convert("RGB")
    else:
        raise ValueError("Entry must contain either 'image_path' or 'image'.")

    img_id = entry["id"]
    input_image_size = image.size

    with torch.no_grad():
        video_frames = pipeline(image, num_frames=frame_count).frames[0]

    resized_frames = [frame.resize(input_image_size) for frame in video_frames]
    
    video_output_path = os.path.join(video_path, f"{img_id}.{format}")

    if format == "gif":
        resized_frames[0].save(
            video_output_path, format='GIF', save_all=True, append_images=resized_frames[1:],
            duration=frame_duration, loop=0
        )
    elif format == "mp4":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 1000 / frame_duration if frame_duration > 0 else 30
        videoWriter = cv2.VideoWriter(video_output_path, fourcc, fps, input_image_size)
        for frame in resized_frames:
            frame_np = np.array(frame, dtype=np.uint8)
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            videoWriter.write(frame_np)
        videoWriter.release()
    
    entry["video_path"] = video_output_path
    
    if "image" in entry:
        del entry["image"]
    
    return entry

def main(args):
    # 加载数据
    if args.input_path:
        logging.info(f"从 JSON 文件加载数据: {args.input_path}")
        data = load_data_from_json(args.input_path)
    else:
        logging.error("必须提供输入路径.")
        return
    
    # 加载模型，并使用 accelerate 处理多进程并发推理
    distributed_state = PartialState()
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", 
        torch_dtype=torch.bfloat16 if args.use_bfloat16 else torch.float16  # 支持bf16
    )
    pipeline.to(distributed_state.device)
    
    # 确保输出目录存在
    os.makedirs(args.video_path, exist_ok=True)
    
    # 分配任务到各个进程
    processed_entries = []
    data_to_process = data[:args.sample_size] if args.debug else data

    with distributed_state.split_between_processes(data_to_process) as entries_per_process:
        for entry in tqdm(entries_per_process, desc="Processing videos"):
            result = process_video(entry, pipeline, args.video_path, args.frame, args.duration, args.format)
            if result:
                processed_entries.append(result)
            with open(args.output_path, 'w') as f:
                json.dump(processed_entries, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch video generation using Stable Diffusion with accelerate.")
    parser.add_argument("--input_path", type=str, help="Path to the input JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--video_path", type=str, required=True, help="Directory to save generated videos.")
    parser.add_argument("--frame", type=int, default=16, help="Number of frames to generate for each video.")
    parser.add_argument("--duration", type=int, default=100, help="Duration for each frame in the generated GIF.")
    parser.add_argument("--debug", action="store_true", help="Process only a subset of entries for testing.")
    parser.add_argument("--sample_size", type=int, default=2, help="Number of entries to process in debug mode.")
    parser.add_argument("--use_bfloat16", action="store_true", help="Use bfloat16 precision for model inference.")
    parser.add_argument("--format", type=str, choices=['gif', 'mp4'], default='mp4', help="Output video format (gif or mp4).")
    args = parser.parse_args()

    main(args)
