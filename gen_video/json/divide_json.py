import json
import os
import math

def load_json(input_file):
    with open(input_file, 'r') as file:
        data = json.load(file)
    return data

def split_list(data, num_parts):
    avg_len = math.ceil(len(data) / num_parts)
    return [data[i:i + avg_len] for i in range(0, len(data), avg_len)]

def save_json_files(data_splits, output_base_path):
    for i, split_data in enumerate(data_splits, start=1):
        output_file = os.path.join(output_base_path, f"output_part_{i}.json")
        with open(output_file, 'w') as file:
            json.dump(split_data, file, indent=2)
        print(f"Saved part {i} to {output_file}")

def main(input_file, output_base_path):
    data = load_json(input_file)
    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list")

    data_splits = split_list(data, 10)
    save_json_files(data_splits, output_base_path)

if __name__ == "__main__":
    input_file = "/home/wenhao/Project/greatxue/MJ-Bench-2.0/gen_video/json/json_backup/input.json"  
    output_base_path = "/home/wenhao/Project/greatxue/MJ-Bench-2.0/gen_video/json/json_backup/new"  
    
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    main(input_file, output_base_path)