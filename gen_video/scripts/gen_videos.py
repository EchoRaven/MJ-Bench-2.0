import argparse
import json
import os
import subprocess
from pathlib import Path
import time
import hashlib

# 可用的模型列表
available_models = ["opensora", "vader", "text_video_diffusion", "instruct_video"]

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate videos based on input prompts using different models."
    )
    parser.add_argument("--input_file_path", required=True, help="Path to the input JSON file")
    parser.add_argument("--output_file_path", required=True, help="Path to save the output JSON file")
    parser.add_argument("--model_list", required=True, help="Comma-separated list of models to use")
    parser.add_argument("--video_base_path", default="./videos", help="Base path for video storage")

    args = parser.parse_args()
    args.model_list = args.model_list.split(',')
    return args

# 验证模型是否有效
def validate_models(model_list):
    for model in model_list:
        if model not in available_models:
            raise ValueError(f"Error: Invalid model '{model}' specified. Available models: {', '.join(available_models)}")

# 生成唯一的视频文件名，并包含序号
def generate_unique_filename(base_path, model, prompt, index):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:4]
    filename = f"{model}_{index}_{prompt_hash}_{timestamp}.mp4"
    return os.path.join(base_path, filename)

# 针对某个模型生成所有提示词的视频
def generate_videos_for_model(prompts, model, video_base_path):
    output_json_body = []
    model_output_dir = os.path.join(video_base_path, model)
    Path(model_output_dir).mkdir(parents=True, exist_ok=True)  # 为模型创建子目录

    for idx, prompt in enumerate(prompts, start=1):  # 使用enumerate添加序号
        video_path = generate_unique_filename(model_output_dir, model, prompt, idx)

        # 根据模型名称调用不同的脚本
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

        # 将生成的视频路径和模型名称保存到输出中
        output_json_body.append({"model": model, "prompt": prompt, "video_path": video_path})

    return output_json_body

# 处理 JSON 文件并按模型顺序生成视频
def process_json_for_models(input_file, model_list, video_base_path, output_file):
    with open(input_file, 'r') as infile:
        prompts = json.load(infile)

    # 验证 JSON 文件是否是列表格式
    if not isinstance(prompts, list):
        raise ValueError("Input JSON must be a list of prompts.")

    output_data = []

    # 按模型顺序生成视频
    for model in model_list:
        print(f"Starting video generation for model: {model}")
        model_output = generate_videos_for_model(prompts, model, video_base_path)
        output_data.extend(model_output)

    # 将生成的视频信息保存到输出文件
    with open(output_file, 'w') as outfile:
        json.dump(output_data, outfile, indent=2)

    print(f"Video generation complete. Output written to: {output_file}")

# 主函数
def main():
    args = parse_args()

    # 验证模型列表是否有效
    validate_models(args.model_list)

    # 根据模型顺序生成视频
    process_json_for_models(args.input_file_path, args.model_list, args.video_base_path, args.output_file_path)

if __name__ == "__main__":
    main()