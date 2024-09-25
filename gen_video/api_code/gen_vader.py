import argparse
import os
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys

def generate_video(prompt, output_path):
    print(f"Generating video for prompt: {prompt} using vader...")
    command = ['python', 'scripts/main/train_t2v_lora.py', 
               '--prompt_fn', 'custom_prompt',
               '--project_dir', 'output_path',
               ]
    
    try:
        # 使用 subprocess.run() 来执行命令
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # 输出执行的标准输出和错误输出
        print("Command output:", result.stdout)
        print("Command errors (if any):", result.stderr)
    
    except subprocess.CalledProcessError as e:
        # 如果命令执行失败，捕获异常并打印错误信息
        print(f"An error occurred while running the command: {e}")
        print(f"Error output: {e.stderr}")
    print(f'Video saved at: {output_video_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate video using text-to-video-synthesis')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for video generation')
    parser.add_argument('--output', type=str, required=True, help='Output video file path')

    args = parser.parse_args()
    
    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generate_video(args.prompt, args.output)