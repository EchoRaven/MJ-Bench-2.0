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
import logging
import importlib.util


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def load_data_from_script(dataset_loader, args):
    spec = importlib.util.spec_from_file_location("dataset_loader", dataset_loader)
    dataset_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset_module)
    
    if hasattr(dataset_module, 'process_dataset'):
        return dataset_module.process_dataset(args.split, args.start_index, args.percentage)
    else:
        raise ImportError("指定的数据加载脚本中不包含 'process_dataset' 函数。")
    
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
    # Check if output file exists to resume from progress
    processed_entries = []
    if os.path.exists(args.output_path):
        logging.info(f"加载现有输出文件 {args.output_path}")
        with open(args.output_path, 'r') as f:
            processed_entries = json.load(f)

    processed_ids = {entry['id'] for entry in processed_entries}

    if args.dataset_loader:
        logging.info(f"使用数据加载器加载数据: {args.dataset_loader}")
        data = load_data_from_script(args.dataset_loader, args)
    else:
        logging.info(f"从 JSON 文件加载数据: {args.input_path}")
        data = load_data_from_json(args.input_path)

    # Skip entries that are already processed
    data_to_process = [entry for entry in data if entry['id'] not in processed_ids]

    if args.debug:
        logging.info(f"Debug模式启用，处理前 {args.sample_size} 条数据")
        data_to_process = data_to_process[:args.sample_size]

    os.makedirs(args.video_path, exist_ok=True)

    logging.info(f"开始并行处理视频生成任务...")

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
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory to store cached models.")
    parser.add_argument("--format", type=str, choices=['gif', 'mp4'], default='mp4', help="Output video format (gif or mp4).")
    parser.add_argument("--dataset_loader", type=str, help="Path to the dataset loader script for loading data from Hugging Face.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to process ('train', 'test', etc.).")
    parser.add_argument("--start_index", type=int, help="Starting index for processing the dataset.")
    parser.add_argument("--percentage", type=float, help="Percentage of the dataset to process starting from start_index.")
    
    args = parser.parse_args()

    main(args)
