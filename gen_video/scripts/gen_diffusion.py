import argparse
import json
import os
import subprocess
from pathlib import Path
import time
import hashlib

available_models = ["opensora", "vader", "text_video_diffusion", "instruct_video"]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate videos based on input prompts using different models."
    )
    parser.add_argument("--input_file_path", required=True, help="Path to the input JSON file")
    parser.add_argument("--output_file_path", required=True, help="Path to save the output JSON file")
    parser.add_argument("--model_list", required=True, help="Comma-separated list of models to use")
    parser.add_argument("--video_base_path", default="./videos4", help="Base path for video storage")

    args = parser.parse_args()
    args.model_list = args.model_list.split(',')
    return args

def validate_models(model_list):
    for model in model_list:
        if model not in available_models:
            raise ValueError(f"Error: Invalid model '{model}' specified. Available models: {', '.join(available_models)}")

def generate_unique_filename(base_path, model, prompt, index):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:4]
    filename = f"{model}_{index}_{prompt_hash}_{timestamp}.mp4"
    return os.path.join(base_path, filename)

def generate_videos_for_model(prompts, model, video_base_path):
    output_json_body = []
    model_output_dir = os.path.join(video_base_path, model)
    Path(model_output_dir).mkdir(parents=True, exist_ok=True)  

    for idx, prompt in enumerate(prompts, start=1):  
        video_path = generate_unique_filename(model_output_dir, model, prompt, idx)
        if model == "opensora":
            print(f"Running Open-Sora for prompt: {prompt}")
            subprocess.run(["python3", "api_code/gen_opensora.py", "--prompt", prompt, "--output", video_path])
        elif model == "vader":
            print(f"Running VADER for prompt: {prompt}")
            subprocess.run(["python3", "api_code/gen_vader.py", "--prompt", prompt, "--output", video_path])
        elif model == "text_video_diffusion":
            print(f"Running Text-Video Diffusion for prompt: {prompt}")
            subprocess.run(["python3", "api_code/gen_text_video_diffusion.py", "--prompt", prompt, "--output", video_path])
        elif model == "instruct_video":
            print(f"Running InstructVideo for prompt: {prompt}")
            subprocess.run(["python3", "api_code/gen_instruct_video.py", "--prompt", prompt, "--output", video_path])
        output_json_body.append({"model": model, "prompt": prompt, "video_path": video_path})
    return output_json_body

def process_json_for_models(input_file, model_list, video_base_path, output_file):
    with open(input_file, 'r') as infile:
        prompts = json.load(infile)
    if not isinstance(prompts, list):
        raise ValueError("Input JSON must be a list of prompts.")

    output_data = []
    for model in model_list:
        print(f"Starting video generation for model: {model}")
        model_output = generate_videos_for_model(prompts, model, video_base_path)
        output_data.extend(model_output)

    with open(output_file, 'w') as outfile:
        json.dump(output_data, outfile, indent=2)

    print(f"Video generation complete. Output written to: {output_file}")

def main():
    args = parse_args()
    validate_models(args.model_list)
    process_json_for_models(args.input_file_path, args.model_list, args.video_base_path, args.output_file_path)

if __name__ == "__main__":
    main()