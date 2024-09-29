import json

with open("hdpv2_2_train.json", "r", encoding="utf-8") as f:
    caption = json.load(f)


new_data = []
with open("hdpv2_2.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    idx = 0
    for c in caption:
        video_0 = data[idx]
        video_1 = data[idx + 1]
        idx += 2
        new_data.append(
            {
                "pair_id": c["id"],
                "video_0": {
                    "video_text": c["caption"],
                    "video_path": video_0["video_path"]
                },
                "video_1": {
                    "video_text": c["caption"],
                    "video_path": video_1["video_path"]
                }
            }
        )

with open("HDPv2_2_train_config.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)