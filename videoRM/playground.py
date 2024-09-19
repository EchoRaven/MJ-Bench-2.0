import json


def convert_format(data, request_id):
    return {
        "id": f"request-{request_id}",
        "source": "safesora/test",
        "caption": data["caption"],
        "subset": data["subset"],
        "video0_body": {
            "video_path": data["video_0_uid"],
            "chosen": data["video_0_issues"]["chosen"],
            "label": data["video_0_issues"]["label"]  # 保持原来的label结构
        },
        "video1_body": {
            "video_path": data["video_1_uid"],
            "chosen": data["video_1_issues"]["chosen"],
            "label": data["video_1_issues"]["label"]  # 保持原来的label结构
        }
    }


def process_multiline_jsonl(input_file, output_file):
    results = []
    buffer = ""

    with open(input_file, 'r', encoding='utf-8') as infile:
        for idx, line in enumerate(infile):
            line = line.strip()
            if line:  # 跳过空行
                buffer += line  # 将所有行加入缓冲区

                # 尝试解析 JSON
                try:
                    data = json.loads(buffer)
                    # 如果解析成功，转换格式
                    converted_data = convert_format(data, len(results))
                    results.append(converted_data)
                    buffer = ""  # 重置缓冲区
                except json.JSONDecodeError:
                    # 如果解析失败，说明 JSON 尚不完整，继续读取下一行
                    continue

    # 将结果列表写入JSON文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, indent=4, ensure_ascii=False)


# 指定输入和输出文件路径
input_jsonl = 'results1.jsonl'  # 输入的jsonl文件路径
output_json = 'safesora_test.json'  # 输出的json文件路径

# 处理多行JSONL文件并生成JSON文件
process_multiline_jsonl(input_jsonl, output_json)

print(f"转换完成！结果已保存到 {output_json}")
