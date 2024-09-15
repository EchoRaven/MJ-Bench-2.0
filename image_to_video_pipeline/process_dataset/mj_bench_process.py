import logging
from datasets import load_dataset
from tqdm import tqdm

def process_dataset(args):
    """
    处理 MJ-Bench 数据集，将 image0 和 image1 转换为视频，并记录 caption, label, 和图像来源。
    """
    # 配置日志输出
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # 加载 Hugging Face 数据集
    hf_token = "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"
    cache_dir = "/remote_shome/snl/feilong/xiapeng/haibo/videoRM/dataset/MJ-Bench"
    dataset = load_dataset("MJ-Bench/MJ-Bench", token=hf_token, cache_dir=cache_dir)

    # 初始化一个空列表，用于存储处理后的数据
    processed_data = []

    # 遍历每个子数据集（alignment, safety, quality, bias）
    for subset_name, subset in dataset.items():
        logging.info(f"Processing subset: {subset_name}, number of rows: {len(subset)}")

        # 使用 tqdm 来显示处理进度
        for i, example in enumerate(tqdm(subset, desc=f"Processing {subset_name}")):
            # 获取 image0 和 image1 作为 PIL Image 对象
            image0 = example['image0']  # 已经是 PIL.Image 对象
            image1 = example['image1']  # 已经是 PIL.Image 对象

            # 记录 image0 的信息
            processed_data.append({
                "id": f"{subset_name}_image0_{i}",
                "image": image0,  # 直接使用 PIL.Image 对象
                "caption": example['caption'],
                "label": example['label'],
                "source": f"{subset_name}/image0"
            })

            # 记录 image1 的信息
            processed_data.append({
                "id": f"{subset_name}_image1_{i}",
                "image": image1,  # 直接使用 PIL.Image 对象
                "caption": example['caption'],
                "label": example['label'],
                "source": f"{subset_name}/image1"
            })
        
        logging.info(f"Finished processing subset: {subset_name}")

    logging.info(f"Processing completed for all subsets. Total items processed: {len(processed_data)}")
    
    return processed_data
