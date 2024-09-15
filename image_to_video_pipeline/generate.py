import os
import json
import argparse
from diffusers import DiffusionPipeline
from PIL import Image
import torch
from torch.multiprocessing import Pool, set_start_method, Manager
from tqdm import tqdm
import cv2
import numpy as np
import logging
import importlib.util
import subprocess
import time
import gc


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
gpu_manager = Manager()
gpu_usage_status = gpu_manager.dict({i: 0 for i in range(torch.cuda.device_count())})

def get_gpu_memory_usage():
    """Returns a list of GPU memory usage as a percentage."""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE,
        encoding='utf-8'
    )
    lines = result.stdout.strip().split('\n')
    memory_usage = []
    for line in lines:
        used, total = map(int, line.split(','))
        memory_usage.append(used / total * 100)  # Calculate percentage usage
    return memory_usage


def get_available_gpu(threshold=30):
    """Returns the ID of the first available GPU with memory usage below the threshold."""
    memory_usage = get_gpu_memory_usage()
    for i, usage in enumerate(memory_usage):
        if usage < threshold and gpu_usage_status[i] == 0:  # 检查是否已经有任务分配
            gpu_usage_status[i] = 1  # 标记该 GPU 正在使用
            return i
    return None


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


def worker(entry, video_path, frame_count, frame_duration, cache_dir, format, max_retries=5):
    retries = 0
    gpu_id = None
    while retries < max_retries:
        if gpu_id is None:
            gpu_id = get_available_gpu(threshold=30)  # Check for an available GPU
        if gpu_id is not None:
            device = torch.device(f"cuda:{gpu_id}")
            try:
                logging.info(f"分配任务给 GPU {gpu_id}，处理任务 {entry['id']}")
                pipeline = DiffusionPipeline.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid-xt", cache_dir=cache_dir
                )
                pipeline = pipeline.to(device)
                torch.cuda.set_per_process_memory_fraction(0.9, device=gpu_id)

                result = process_video(entry, pipeline, video_path, frame_count, frame_duration, format)
                
                # 任务完成后，释放 GPU 占用资源
                del pipeline  # 删除 pipeline 对象
                torch.cuda.empty_cache()  # 释放 GPU 缓存
                gc.collect()  # 强制运行垃圾回收

                gpu_usage_status[gpu_id] = 0  # 任务完成后释放 GPU

                # 输出任务完成信息
                logging.info(f"任务 {entry['id']} 已完成，使用 GPU {gpu_id}")

                return result
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    logging.warning(f"GPU {gpu_id} 内存不足，清理后重试...")
                    retries += 1
                    time.sleep(5)  # Wait and retry after clearing memory
                else:
                    logging.error(f"GPU {gpu_id} 出现错误: {e}")
                    gpu_usage_status[gpu_id] = 0  # 任务失败后释放 GPU
                    return None
        else:
            logging.info("所有GPU繁忙，等待中...")
            time.sleep(5)  # Wait and retry after 5 seconds
    
    logging.error(f"任务 {entry['id']} 在 {max_retries} 次重试后失败")
    return None  # Return None if all retries fail


def load_data_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def load_data_from_script(dataset_loader, args):
    spec = importlib.util.spec_from_file_location("dataset_loader", dataset_loader)
    dataset_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset_module)
    
    if hasattr(dataset_module, 'process_dataset'):
        return dataset_module.process_dataset(args)
    else:
        raise ImportError("指定的数据加载脚本中不包含 'process_dataset' 函数。")


def main(args):
    if args.dataset_loader:
        logging.info(f"使用数据加载器加载数据: {args.dataset_loader}")
        data = load_data_from_script(args.dataset_loader, args)
    else:
        logging.info(f"从 JSON 文件加载数据: {args.input_path}")
        data = load_data_from_json(args.input_path)

    if args.debug:
        logging.info(f"Debug模式启用，处理前 {args.sample_size} 条数据")
        data = data[:args.sample_size]

    os.makedirs(args.video_path, exist_ok=True)

    logging.info(f"开始并行处理视频生成任务...")

    pool = Pool(processes=torch.cuda.device_count())

    tasks = []
    for entry in data:
        tasks.append(pool.apply_async(worker, (entry, args.video_path, args.frame, args.duration, args.cache_dir, args.format)))

    results = [task.get() for task in tqdm(tasks, desc="Processing videos")]

    # Remove None results (failed tasks)
    results = [result for result in results if result is not None]

    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=4)

    logging.info(f"结果已保存到 {args.output_path}")


if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Batch video generation from images using Stable Diffusion.")
    parser.add_argument("--input_path", type=str, help="Path to the input JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--video_path", type=str, required=True, help="Directory to save generated videos.")
    parser.add_argument("--frame", type=int, default=16, help="Number of frames to generate for each video.")
    parser.add_argument("--duration", type=int, default=100, help="Duration for each frame in the generated GIF.")
    parser.add_argument("--debug", action="store_true", help="Process only a subset of entries for testing.")
    parser.add_argument("--sample_size", type=int, default=2, help="Number of entries to process in debug mode.")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory to store cached models.")
    parser.add_argument("--format", type=str, choices=['gif', 'mp4'], default='mp4', help="Output video format (gif or mp4).")
    parser.add_argument("--dataset_loader", type=str, help="Path to the dataset loader script for loading data from Hugging Face.")

    args = parser.parse_args()
    main(args)
