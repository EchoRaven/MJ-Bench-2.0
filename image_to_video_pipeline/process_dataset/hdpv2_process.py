import logging
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import os
import ast
from PIL import UnidentifiedImageError
from io import BytesIO

def process_dataset(args):
    """
    处理 HPDv2 数据集，将 image_path 中的图像加载为 PIL Image，并记录 prompt, human_preference, 和图像路径。
    处理所有 splits，包括 train, validation, 和 test。
    """
    # 配置日志输出
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # 加载 Hugging Face 数据集
    hf_token = "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"
    cache_dir = "/work/vita/nie/haibo/image_to_video_pipeline/dataset/HPDv2"

    dataset = load_dataset("ymhao/HPDv2", token=hf_token, cache_dir=cache_dir, trust_remote_code=True)
    processed_data = []
    # 遍历数据集的每个 split
    # print(dataset)
    for split in dataset.keys():
        logging.info(f"Processing {split} split of HPDv2 dataset")
        # print(dataset['train'][10214])
        # 遍历当前 split 的数据
        length = len(dataset[split])
        for i in tqdm(range(length), desc=f"Processing {split} split"):
            try:
                example = dataset[split][i]
                prompt = example['prompt']
                human_preference = example['human_preference']
                image_paths = example['image_path']
                images = example['image']
                # images[0] 和 images[1] 已经是 PIL.Image 对象
                image0 = images[0]  # 直接获取 PIL.Image 对象
                processed_data.append({
                    "id": f"HPDv2_{split}_image0_{i}",
                    "image": image0,
                    "prompt": prompt,
                    "label": human_preference[0],
                    "source": f"HPDv2/{split}"
                })

                image1 = images[1]  # 直接获取 PIL.Image 对象
                processed_data.append({
                    "id": f"HPDv2_{split}_image1_{i}",
                    "image": image1,
                    "prompt": prompt,
                    "label": human_preference[0],
                    "source": f"HPDv2/{split}"
                })

            except Exception as e:
                # Log the error and skip the problematic example
                logging.warning(f"Skipping example at index {i} in {split} split due to error: {e}")
                continue

    logging.info(f"Processing completed. Total items processed: {len(processed_data)}")
    
    return processed_data
