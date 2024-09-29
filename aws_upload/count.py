import os
import json

directory = "label_files"

entry_dict = {}

for root, dirs, files in os.walk(directory):
    for file_name in files:
        file_path = os.path.join(root, file_name)  # 获取完整文件路径
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for d in data:
                subset = d["subset"]
                if subset not in entry_dict.keys():
                    entry_dict[subset] = []
                entry_dict[subset].append(d)

save_dir = "subset"

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for key in entry_dict.keys():
    with open(f"{save_dir}/{key}.json", "w", encoding="utf-8") as f:
        json.dump(entry_dict[key], f, ensure_ascii=False, indent=4)

