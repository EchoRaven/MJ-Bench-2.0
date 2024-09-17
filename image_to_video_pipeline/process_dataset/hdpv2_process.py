import logging
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import os

def process_hpdv2_dataset(args):
    """
    处理 HPDv2 数据集，将 image_path 中的图像加载为 PIL Image，并记录 prompt, human_preference, 和图像路径。
    处理所有 splits，包括 train, validation, 和 test。
    """
    # 配置日志输出
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # 加载 Hugging Face 数据集
    hf_token = "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"
    cache_dir = "/remote_shome/snl/feilong/xiapeng/haibo/videoRM/dataset/HPDv2"

    dataset = load_dataset("ymhao/HPDv2", token=hf_token, cache_dir=cache_dir)

    # 初始化一个空列表，用于存储所有 splits 的处理数据
    all_processed_data = {}

    # 遍历数据集的每个 split
    for split in dataset.keys():
        logging.info(f"Processing {split} split of HPDv2 dataset")

        # 初始化一个空列表，用于存储当前 split 的处理数据
        processed_data = []

        # 遍历当前 split 的数据
        for i, example in enumerate(tqdm(dataset[split], desc=f"Processing {split} split")):
            prompt = example['prompt']
            human_preference = example['human_preference']
            image_paths = example['image_path']

            # 读取并处理 image0
            image0_path = image_paths[0]
            image0 = Image.open(image0_path)

            processed_data.append({
                "id": f"HPDv2_{split}_image0_{i}",
                "image": image0,  # 直接使用 PIL.Image 对象
                "prompt": prompt,
                "label": human_preference[0],
                "source": f"HPDv2/{split}"
            })

            # 读取并处理 image1
            image1_path = image_paths[1]
            image1 = Image.open(image1_path)

            processed_data.append({
                "id": f"HPDv2_{split}_image1_{i}",
                "image": image1,  # 直接使用 PIL.Image 对象
                "prompt": prompt,
                "label": human_preference[0],
                "source": f"HPDv2/{split}"
            })
        
        # 将处理后的数据存储在字典中，以 split 名称为键
        all_processed_data[split] = processed_data

        logging.info(f"Processing completed for {split} split. Total items processed: {len(processed_data)}")
    
    return all_processed_data
