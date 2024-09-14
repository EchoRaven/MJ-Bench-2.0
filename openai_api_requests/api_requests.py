import argparse
import json
import os
import yaml
import time
import random
import concurrent.futures
from utils.openai import load_client
from openai import OpenAIError


# 加载API客户端
def load_openai_client(config_path):
    return load_client(config_path)

# 处理单个请求，返回结果
def process_request(client, request_data, max_retries=3):
    attempt = 0
    while attempt < max_retries:
        try:
            chat_completion = client.chat.completions.create(
                model=request_data['body']['model_type'],
                messages=[
                    {"role": "system", "content": request_data['body']['system_prompt']},
                    {"role": "user", "content": request_data['body']['user_prompt']}
                ],
                max_tokens=request_data['body']['max_token']
            )
            return {"id": request_data['id'], "response": chat_completion.choices[0].message.content}
        except OpenAIError as e:
            print(f"Error on request {request_data['id']}: {str(e)}. Retrying {attempt + 1}/{max_retries}...")
            attempt += 1
            time.sleep(2)  # 等待2秒后重试
    return {"id": request_data['id'], "response": None}

# 保存结果到文件
def save_partial_results(output_file, results):
    with open(output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

# 并发处理函数
def process_requests_concurrently(client, input_data, output_file, max_workers, max_retries):
    results = []
    
    # 如果输出文件已存在，读取已保存的结果，避免重复处理
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)
            completed_ids = {result["id"] for result in results}
            input_data = [req for req in input_data if req["id"] not in completed_ids]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {executor.submit(process_request, client, req, max_retries): req["id"] for req in input_data}
        for future in concurrent.futures.as_completed(future_to_id):
            try:
                result = future.result()
                if result:
                    results.append(result)
                    save_partial_results(output_file, results)  # 每次请求完成后立即保存
            except Exception as e:
                print(f"Request failed with exception: {e}")
    
    print(f"All requests processed. Results saved to {output_file}.")

# 主函数
def main():
    parser = argparse.ArgumentParser(description="Concurrent OpenAI API requests handler")
    parser.add_argument('--input_file', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output JSON file')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file with API key')
    parser.add_argument('--max_workers', type=int, default=5, help='Max number of concurrent requests')
    parser.add_argument('--max_retries', type=int, default=3, help='Max retries for a single request')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to sample data')
    parser.add_argument('--sample_size', type=int, default=10, help='Number of samples to process in debug mode')

    args = parser.parse_args()

    # 加载OpenAI客户端
    client = load_openai_client(args.config)

    # 读取输入文件
    with open(args.input_file, 'r', encoding="utf-8") as f:
        input_data = json.load(f)

    # 如果启用了调试模式，进行数据抽样
    if args.debug:
        print(f"Debug mode enabled. Sampling {args.sample_size} requests from the input data.")
        if len(input_data) > args.sample_size:
            input_data = random.sample(input_data, args.sample_size)
        else:
            print(f"Input data contains fewer than {args.sample_size} items. Processing all data.")

    # 并发处理请求
    process_requests_concurrently(client, input_data, args.output_file, args.max_workers, args.max_retries)

if __name__ == "__main__":
    main()
