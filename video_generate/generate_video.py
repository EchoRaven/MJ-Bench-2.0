import argparse
import json
import os


def generate_video(prompt, model_name, video_base_path):
    #TODO
    video_path = os.path.join(video_base_path, f"{model_name}_{hash(prompt)}.mp4")
    print(f"Generating video for prompt '{prompt}' using model '{model_name}'")
    return video_path

def read_input_file(input_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def write_output_file(output_file_path, data):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def process_data(data, model_list, video_base_path):
    output_data = []
    for item in data:
        if isinstance(item, dict):
            prompt = item.get("text", "")
        else:
            prompt = item
        
        body = []
        for model_name in model_list:
            video_path = generate_video(prompt, model_name, video_base_path)
            body.append({
                "model": model_name,
                "video_path": video_path
            })
        
        output_data.append({
            "text": prompt,
            "body": body
        })
    return output_data

def main():
    parser = argparse.ArgumentParser(description='Video Generation Script')
    parser.add_argument('--input_file_path', type=str, required=True)
    parser.add_argument('--output_file_path', type=str, required=True)
    parser.add_argument('--model_list', type=str, required=True)
    parser.add_argument('--video_base_path', type=str, default='./videos')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    model_list = args.model_list.split(',')
    available_models = ['opensora', 'vader', 'videocrafter', 'stable_video_diffusion', 'modelscope_text_video_diffusion', 'InstructVideo']
    
    for model_name in model_list:
        if model_name not in available_models:
            raise ValueError(f"Models with error: {model_name}. Models available: {available_models}")

    input_data = read_input_file(args.input_file_path)
    output_data = process_data(input_data, model_list, args.video_base_path)
    
    write_output_file(args.output_file_path, output_data)

if __name__ == "__main__":
    main()
