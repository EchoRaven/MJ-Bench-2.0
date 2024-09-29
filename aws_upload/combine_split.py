import os
import json


def merge_json_files(directory, output_file):
    merged_data = []

    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)

            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    # 将数据添加到合并列表中
                    merged_data.extend(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding {filename}: {e}")

    # 将合并的数据写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    print(f'合并完成，结果保存至: {output_file}')


# 使用示例
directory_path = 'download_files'  # 替换为你的JSON文件夹路径
output_file = 'hdpv2_2_train.json'  # 合并后的输出文件

merge_json_files(directory_path, output_file)
