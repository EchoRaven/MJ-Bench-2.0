import logging
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def save_image(image, image_path, i, split, index):
    """
    Save the image to the specified path. If the image already exists, skip saving.
    """
    try:
        if not os.path.exists(image_path):
            image.save(image_path)
        return {
            "id": f"HPDv2_{split}_image{index}_{i}",
            "image_path": image_path,
        }
    except Exception as e:
        logging.warning(f"Error saving image at index {i} in {split} split: {e}")
        return None

def process_dataset(split="test", start_index=0, percentage=20):
    """
    Process the HPDv2 dataset, loading images as PIL images and recording prompt, 
    human_preference, and image paths. Process only a part of the dataset based on 
    percentage and split.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    hf_token = "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"
    cache_dir = "../dataset/HPDv2"
    image_save_dir = os.path.join(cache_dir, "images")
    os.makedirs(image_save_dir, exist_ok=True)

    dataset = load_dataset("ymhao/HPDv2", token=hf_token, cache_dir=cache_dir, trust_remote_code=True)
    processed_data = []

    # Ensure the specified split exists
    if split in dataset.keys():
        logging.info(f"Processing {split} split of HPDv2 dataset")
        total_length = len(dataset[split])

        # Determine the range of the dataset to process
        split_length = int(total_length * (percentage / 100))
        end_index = min(start_index + split_length, total_length)
        logging.info(f"Processing from index {start_index} to {end_index} (total {end_index - start_index} examples)")

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in tqdm(range(start_index, end_index), desc=f"Processing {split} split"):
                try:
                    example = dataset[split][i]
                    prompt = example['prompt']
                    human_preference = example['human_preference']
                    images = example['image']

                    # Submit the first image to the thread pool for saving
                    image_path0 = os.path.join(image_save_dir, f"HPDv2_{split}_image0_{i}.png")
                    futures.append(executor.submit(save_image, images[0], image_path0, i, split, 0))

                    # Submit the second image to the thread pool for saving
                    image_path1 = os.path.join(image_save_dir, f"HPDv2_{split}_image1_{i}.png")
                    futures.append(executor.submit(save_image, images[1], image_path1, i, split, 1))

                    # Add metadata for the images
                    for j in [0, 1]:
                        processed_data.append({
                            "id": f"HPDv2_{split}_image{j}_{i}",
                            "prompt": prompt,
                            "label": human_preference[0],
                            "source": f"HPDv2/{split}"
                        })

                except Exception as e:
                    logging.warning(f"Skipping example at index {i} in {split} split due to error: {e}")
                    continue

            # Collect all completed image save tasks
            for future in as_completed(futures):
                result = future.result()
                if result:
                    # Update the processed_data with the saved image path
                    for data in processed_data:
                        if data["id"] == result["id"]:
                            data["image_path"] = result["image_path"]

    logging.info(f"Processing completed. Total items processed: {len(processed_data)}")

    return processed_data
