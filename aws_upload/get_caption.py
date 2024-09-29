import json

with open("hdpv2_2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

new_data = []

for d in data:
    label = d["label"]
    if f"{label}" == d["id"].split("_")[2][-1]:
        new_data.append(d)

with open("hdpv2_2_c.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)