import argparse
import os
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def generate_video(prompt, output_path):
    print(f"Generating video for prompt: {prompt} using text-to-video-synthesis...")
    p = pipeline('text-to-video-synthesis', 'damo/text-to-video-synthesis')
    
    test_text = {
        'text': prompt,
    }
    
    output_video_path = p(test_text, output_video=output_path)[OutputKeys.OUTPUT_VIDEO]
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

