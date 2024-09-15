import os
import json
import argparse
from diffusers import DiffusionPipeline
from PIL import Image
import torch
from torch.multiprocessing import Pool, set_start_method
from tqdm import tqdm
import cv2
import numpy as np


def process_video(entry, pipeline, video_path, frame_count, frame_duration, format):
    """Process each entry to generate video and save it."""
    image_path = entry["image_path"]
    img_id = entry["id"]
    
    # Load image
    input_image = Image.open(image_path).convert("RGB")
    input_image_size = input_image.size
    
    # Generate video frames
    with torch.no_grad():
        video_frames = pipeline(input_image, num_frames=frame_count).frames[0]
    
    # Resize frames to match input image size
    resized_frames = [frame.resize(input_image_size) for frame in video_frames]
    if format == "gif":
        # Create video output path
        video_output_path = os.path.join(video_path, f"{img_id}.gif")
        
        # Save as GIF
        resized_frames[0].save(
            video_output_path, format='GIF', save_all=True, append_images=resized_frames[1:], 
            duration=frame_duration, loop=0
        )
    elif format == "mp4":
        video_output_path = os.path.join(video_path, f"{img_id}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 1000 / frame_duration if frame_duration > 0 else 30  # 如果frame_duration是0，将导致除以0的错误
        videoWriter = cv2.VideoWriter(video_output_path, fourcc, fps, input_image_size)
        for frame in resized_frames:
            # Convert the frame (PIL Image) to a numpy array with uint8 type
            frame_np = np.array(frame, dtype=np.uint8)

            # Convert RGB (used by PIL) to BGR (used by OpenCV)
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

            # Check if the frame is a valid numpy array of uint8 type
            if isinstance(frame_np, np.ndarray) and frame_np.dtype == np.uint8:
                try:
                    videoWriter.write(frame_np)  # Write the frame to the video file
                except Exception as e:
                    print(f"Error writing frame: {e}")
            else:
                print("Frame is not a valid numpy uint8 array")
    
    # Add video path to entry
    entry["video_path"] = video_output_path
    
    return entry


def worker(entry, gpu_id, video_path, frame_count, frame_duration, cache_dir, format):
    """Function that each worker will run in parallel, loading the pipeline on a specific GPU."""
    # Set device to the given GPU
    device = torch.device(f"cuda:{gpu_id}")
    
    # Load pipeline for this worker
    try:
        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt", cache_dir=cache_dir
        )
        pipeline = pipeline.to(device)
        
        # Monitor GPU memory usage
        torch.cuda.set_per_process_memory_fraction(0.9, device=gpu_id)  # Limit each process to 90% of the GPU memory
        
        return process_video(entry, pipeline, video_path, frame_count, frame_duration, format)
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            print(f"GPU {gpu_id} is out of memory, retrying...")
            return worker(entry, gpu_id, video_path, frame_count, frame_duration, cache_dir, format)
        else:
            raise e


def main(args):
    # Load input data from JSON
    with open(args.input_path, 'r') as f:
        data = json.load(f)
    
    # If in debug mode, sample 2 entries
    if args.debug:
        data = data[:2]
    
    # Create video output directory if it doesn't exist
    os.makedirs(args.video_path, exist_ok=True)
    
    # Multi-GPU processing
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs. Starting parallel processing...")

    # Create a pool of workers
    pool = Pool(processes=num_gpus)
    
    tasks = []
    
    # Distribute tasks across GPUs without using Queue
    for i, entry in enumerate(data):
        gpu_id = i % num_gpus
        tasks.append(pool.apply_async(worker, (entry, gpu_id, args.video_path, args.frame, args.duration, args.cache_dir, args.format)))
    
    # Collect results
    results = [task.get() for task in tqdm(tasks, desc="Processing videos")]
    
    # Save the results to the output JSON file
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    # Setup multiprocessing
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    
    # Argument parsing
    parser = argparse.ArgumentParser(description="Batch video generation from images using Stable Diffusion.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--video_path", type=str, required=True, help="Directory to save generated videos.")
    parser.add_argument("--frame", type=int, default=16, help="Number of frames to generate for each video.")
    parser.add_argument("--duration", type=int, default=100, help="Duration for each frame in the generated GIF.")
    parser.add_argument("--debug", action="store_true", help="Process only two entries for testing.")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory to store cached models.")
    parser.add_argument("--format", type=str, choices=['gif', 'mp4'], default='mp4', help="Output video format (gif or mp4).")

    args = parser.parse_args()

    main(args)
