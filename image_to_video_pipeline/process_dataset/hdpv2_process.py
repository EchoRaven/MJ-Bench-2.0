import logging
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

def process_dataset(args):
    """
    处理 HPDv2 数据集，将 image_path 中的图像加载为 PIL Image，并逐条返回数据。
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    hf_token = "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"
    cache_dir = "/work/vita/nie/haibo/image_to_video_pipeline/dataset/HPDv2"

    dataset = load_dataset("ymhao/HPDv2", token=hf_token, cache_dir=cache_dir, trust_remote_code=True)
    
    for split in dataset.keys():
        logging.info(f"Processing {split} split of HPDv2 dataset")
        length = len(dataset[split])
        for i in tqdm(range(length), desc=f"Processing {split} split"):
            try:
                example = dataset[split][i]
                prompt = example['prompt']
                human_preference = example['human_preference']
                images = example['image']
                
                image0 = images[0]  # 获取 PIL.Image 对象
                yield {
                    "id": f"HPDv2_{split}_image0_{i}",
                    "image": image0,
                    "prompt": prompt,
                    "label": human_preference[0],
                    "source": f"HPDv2/{split}"
                }

                image1 = images[1]  # 获取 PIL.Image 对象
                yield {
                    "id": f"HPDv2_{split}_image1_{i}",
                    "image": image1,
                    "prompt": prompt,
                    "label": human_preference[0],
                    "source": f"HPDv2/{split}"
                }

            except Exception as e:
                logging.warning(f"Skipping example at index {i} in {split} split due to error: {e}")
                continue
