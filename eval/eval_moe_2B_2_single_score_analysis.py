import json
import os
import re
import time
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from collections import Counter
import sys
import logging
sys.path.append('../')


from swift.llm import (
    get_model_tokenizer, get_template, inference,ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
import torch

if not os.path.exists('./output'):
    os.mkdir('./output')

from MoE.module import MJ_VIDEO
with open("../MoE/MoE_config_2B_2.json", "r", encoding="utf-8") as f:
    config = json.load(f)
model = MJ_VIDEO(config)
seed_everything(42)


def evaluate_videos(caption, video0_path, video1_path, force_keys=[]):
    logging.info(f"caption : {caption}")
    response0, score0 = model.inference([video0_path], caption, force_keys, "single_video_analysis_score_prompt_template")
    response1, score1 = model.inference([video1_path], caption, force_keys, "single_video_analysis_score_prompt_template")
    logging.info(f"response 0: {response0}")
    logging.info(f"response 1: {response1}")
    chosen = ""
    if score0 > score1:
        chosen = "video 1"
    elif score0 < score1:
        chosen = "video 2"
    else:
        chosen = "same"
    logging.info(f"chosen : {chosen}")
    return chosen, score0, score1


def process_json_file(json_file_path, videos_dir, output_file_name, key):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    all_results = []
    true_labels = []
    predictions = []
    counter = 0

    for item in data:
        try:
            caption = item['caption']
            video0_path_relative = item['video0_body']['video_path']
            video1_path_relative = item['video1_body']['video_path']
            video0_path = os.path.join(videos_dir, video0_path_relative)
            video1_path = os.path.join(videos_dir, video1_path_relative)

            true_chosen = item['video0_body']['chosen']

            chosen, score_1, score_2 = evaluate_videos(caption, video0_path, video1_path)
            if chosen == "same":
                video_0_rating = score_1
                video_1_rating = score_2
                model_chosen = True
            elif chosen == "video 1":
                video_0_rating = score_1
                video_1_rating = score_2
                model_chosen = True
            elif chosen == "video 2":
                video_0_rating = score_1
                video_1_rating = score_2
                model_chosen = False
            else:
                video_0_rating = -1
                video_1_rating = -1
                model_chosen = True

            result = {
                "caption": caption,
                "video_0_uid": video0_path,
                "video_1_uid": video1_path,
                "video_0_scores": {
                    "alignment": video_0_rating
                },
                "video_1_scores": {
                    "alignment": video_1_rating
                },
                "chosen": model_chosen
            }
            all_results.append(result)

            true_labels.append(true_chosen)
            predictions.append(model_chosen)
            counter = counter + 1
            if counter % 10 == 0:
                accuracy = accuracy_score(true_labels, predictions)
                f1 = f1_score(true_labels, predictions)
                recall = recall_score(true_labels, predictions)
                precision = precision_score(true_labels, predictions)
                
                with open(f"./output/moe_2B_2_{key}_single_analysis_score.txt", 'w') as file:
                    file.write(f"Accuracy: {accuracy:.2f}\\n")
                    file.write(f"F1 Score: {f1:.2f}\\n")
                    file.write(f"Recall: {recall:.2f}\\n")
                    file.write(f"Precision: {precision:.2f}\\n")

                print(f"Accuracy: {accuracy:.2f}")
                print(f"F1 Score: {f1:.2f}")
                print(f"Recall: {recall:.2f}")
                print(f"Precision: {precision:.2f}")
                
                output_file = os.path.join('./output',output_file_name)
                with open(output_file, 'w') as outfile:
                    json.dump(all_results, outfile, indent=4)
        except:
            continue

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    
    with open(f"./output/moe_2B_2_{key}_single_analysis_score.txt", 'w') as file:
        file.write(f"Accuracy: {accuracy:.2f}\\n")
        file.write(f"F1 Score: {f1:.2f}\\n")
        file.write(f"Recall: {recall:.2f}\\n")
        file.write(f"Precision: {precision:.2f}\\n")

    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Precision: {precision:.2f}")

    output_file = os.path.join('./output',output_file_name)
    with open(output_file, 'w') as outfile:
        json.dump(all_results, outfile, indent=4)


def process_overall_file(json_file_path, videos_dir, output_file_name):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    all_results = []
    true_labels = []
    predictions = []
    counter = 0

    for item in data:
        try:
            caption = item['caption']
            video0_path_relative = item['chosen']
            video1_path_relative = item['reject']
            video0_path = os.path.join(videos_dir, video0_path_relative)
            video1_path = os.path.join(videos_dir, video1_path_relative)
            true_chosen = True
            chosen, score_1, score_2 = evaluate_videos(caption, video0_path, video1_path)
            if chosen == "same":
                video_0_rating = score_1
                video_1_rating = score_2
                model_chosen = True
            elif chosen == "video 1":
                video_0_rating = score_1
                video_1_rating = score_2
                model_chosen = True
            elif chosen == "video 2":
                video_0_rating = score_1
                video_1_rating = score_2
                model_chosen = False
            else:
                video_0_rating = -1
                video_1_rating = -1
                model_chosen = True

            result = {
                "caption": caption,
                "video_0_uid": video0_path,
                "video_1_uid": video1_path,
                "video_0_scores": {
                    "alignment": video_0_rating
                },
                "video_1_scores": {
                    "alignment": video_1_rating
                },
                "chosen": model_chosen
            }
            all_results.append(result)

            true_labels.append(true_chosen)
            predictions.append(model_chosen)
            counter = counter + 1
            if counter % 10 == 0:
                accuracy = accuracy_score(true_labels, predictions)
                f1 = f1_score(true_labels, predictions)
                recall = recall_score(true_labels, predictions)
                precision = precision_score(true_labels, predictions)
                
                with open(f"./output/moe_2B_2_{key}_single_analysis_score.txt", 'w') as file:
                    file.write(f"Accuracy: {accuracy:.2f}\\n")
                    file.write(f"F1 Score: {f1:.2f}\\n")
                    file.write(f"Recall: {recall:.2f}\\n")
                    file.write(f"Precision: {precision:.2f}\\n")

                print(f"Accuracy: {accuracy:.2f}")
                print(f"F1 Score: {f1:.2f}")
                print(f"Recall: {recall:.2f}")
                print(f"Precision: {precision:.2f}")
                
                output_file = os.path.join('./output',output_file_name)
                with open(output_file, 'w') as outfile:
                    json.dump(all_results, outfile, indent=4)
        except:
            continue

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    
    with open(f"./output/moe_2B_2_{key}_single_analysis_score.txt", 'w') as file:
        file.write(f"Accuracy: {accuracy:.2f}\\n")
        file.write(f"F1 Score: {f1:.2f}\\n")
        file.write(f"Recall: {recall:.2f}\\n")
        file.write(f"Precision: {precision:.2f}\\n")

    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Precision: {precision:.2f}")

    output_file = os.path.join('./output',output_file_name)
    with open(output_file, 'w') as outfile:
        json.dump(all_results, outfile, indent=4)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
videos_dir = '../videos'
json_files = {
    'overall': '../test/overall.json',
    'safety': '../test/safety.json',
    'alignment': '../test/alignment.json',
    'bias_fairness': '../test/bias.json',
    'quality': '../test/quality.json',
    'coherence_consistency': '../test/cc.json',
}

for key, value in json_files.items():
    json_file_path = value
    output_file_name = f'moe_2B_2_{key}_single_analysis_score.json'
    
    # 检查是否为overall文件
    if key == 'overall':
        process_overall_file(json_file_path, videos_dir, output_file_name)  # 使用另一个函数处理
    else:
        process_json_file(json_file_path, videos_dir, output_file_name, key)


























