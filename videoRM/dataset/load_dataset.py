import os
import random
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, load_from_disk
from torchvision.transforms.functional import InterpolationMode


# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VIDEO_PATH_PREFIX = "/remote_shome/snl/feilong/xiapeng/haibo/videoRM/dataset/SafeSora"
IMAGE_SIZE = 448
NUM_IMAGE_TOKEN = 256
QUESTION_PROMPT = "Do you think the video generated as this prompt is of good quality (Yes/No)?"


def build_transform(input_size):
    """Build transformation pipeline for image pre-processing."""
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find closest target aspect ratio based on input image's aspect ratio."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size ** 2 * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=IMAGE_SIZE, use_thumbnail=False):
    """Preprocess image dynamically based on its aspect ratio and target sizes."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1]
    )

    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []

    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_image(image_file, input_size=IMAGE_SIZE, max_num=12):
    """Load and preprocess a single image."""
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values


def get_index(bound, fps, max_frame, num_segments=32, first_idx=0):
    """Get frame indices based on video segmentation."""
    start, end = bound or (-100000, 100000)
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    return np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])


def load_video(video_path, bound=None, input_size=IMAGE_SIZE, max_num=1, num_segments=32):
    """Load video and extract frame pixel values."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    transform = build_transform(input_size=input_size)
    
    frame_indices = get_index(bound, fps, max_frame, num_segments=num_segments)
    pixel_values_list, num_patches_list = [], []

    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        processed_images = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = torch.stack([transform(tile) for tile in processed_images])
        pixel_values_list.append(pixel_values)
        num_patches_list.append(pixel_values.shape[0])

    return torch.cat(pixel_values_list), num_patches_list


def fill_with_image_token(query, num_image_token, num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'):
    """Fill query with image tokens based on patches."""
    for num_patches in num_patches_list:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)
    return query

def create_chosen_list(examples):
    chosen_list = []
    for helpfulness, harmlessness, sub_preferences in zip(examples["helpfulness"], examples["harmlessness"], examples["sub_preferences"]):
        count_0 = 0
        count_1 = 0
        if helpfulness == "video_0":
            count_0 += 1
        else:
            count_1 += 1
        if harmlessness == "video_0":
            count_0 += 1
        else:
            count_1 += 1
        for key in sub_preferences:
            if sub_preferences[key] == "video_0":
                count_0 += 1
            else:
                count_1 += 1
        if count_0 > count_1:
            chosen_list.append(0)
        elif count_0 < count_1:
            chosen_list.append(1)
        else:
            chosen_list.append(random.randint(0, 1))
    return chosen_list

def pad_labels(labels, max_length):
    """Pads the labels list to max_length with -100."""
    return [label + [-100] * (max_length - len(label)) for label in labels]
    
def preprocess_function(examples, tokenizer, max_length=4096, num_image_token=NUM_IMAGE_TOKEN, num_segments=8):
    """Preprocess video examples with text and image tokens, ensuring label consistency after padding."""
    
    # Prepare video paths for both videos
    video_paths_0 = [os.path.join(VIDEO_PATH_PREFIX, info['video_path']) for info in examples['video_0']]
    video_paths_1 = [os.path.join(VIDEO_PATH_PREFIX, info['video_path']) for info in examples['video_1']]

    # Load video frames and extract pixel values for both video sets
    pixel_values_0, num_patches_list_0 = zip(*[load_video(path, num_segments=num_segments, max_num=1) for path in video_paths_0])
    pixel_values_1, num_patches_list_1 = zip(*[load_video(path, num_segments=num_segments, max_num=1) for path in video_paths_1])

    # Create frame-based prefixes for both videos
    video_pre_prefix_list_0 = [''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches))]) for num_patches in num_patches_list_0]
    video_pre_prefix_list_1 = [''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches))]) for num_patches in num_patches_list_1]

    # Fill image tokens in the prefixes
    video_prefix_list_0 = [fill_with_image_token(prefix, num_image_token, num_patch) for num_patch, prefix in zip(num_patches_list_0, video_pre_prefix_list_0)]
    video_prefix_list_1 = [fill_with_image_token(prefix, num_image_token, num_patch) for num_patch, prefix in zip(num_patches_list_1, video_pre_prefix_list_1)]

    # Choose the response video based on the chosen list
    chosen_list = create_chosen_list(examples)

    video_texts_0 = [video_info['video_text'] for video_info in examples['video_0']]
    video_texts_1 = [video_info['video_text'] for video_info in examples['video_1']]

    # Construct final prompt strings
    question_list_0 = [f"{prefix}Prompt: {text}Question: {QUESTION_PROMPT}\nYes" if chosen == 0 else f"Prompt: {text}Question: {QUESTION_PROMPT}\nNo"
                       for prefix, text, chosen in zip(video_prefix_list_0, video_texts_0, chosen_list)]
    question_list_1 = [f"{prefix}Prompt: {text}Question: {QUESTION_PROMPT}\nYes" if chosen == 1 else f"Prompt: {text}Question: {QUESTION_PROMPT}\nNo"
                       for prefix, text, chosen in zip(video_prefix_list_1, video_texts_1, chosen_list)]

    # Tokenize the questions without padding to ensure "Yes" and "No" are at the correct positions
    tokenized_0 = tokenizer(question_list_0, truncation=True, padding=False)
    tokenized_1 = tokenizer(question_list_1, truncation=True, padding=False)

    # Generate labels for each tokenized sequence
    labels_0 = create_labels(tokenized_0['input_ids'], tokenizer)
    labels_1 = create_labels(tokenized_1['input_ids'], tokenizer)

    # Now, pad input_ids, attention_mask, and labels together to max_length
    tokenized_0 = tokenizer(question_list_0, truncation=True, padding='max_length', max_length=max_length)
    tokenized_1 = tokenizer(question_list_1, truncation=True, padding='max_length', max_length=max_length)
    
    # Ensure padding consistency by applying the same padding to labels
    labels_0 = pad_labels(labels_0, max_length)
    labels_1 = pad_labels(labels_1, max_length)

    # Ensure the lengths of input_ids, attention_mask, and labels are consistent across the dataset
    assert len(tokenized_0['input_ids']) == len(tokenized_1['input_ids']), "Tokenized input lengths do not match!"
    assert len(labels_0) == len(labels_1), "Labels lengths do not match!"

    # Return the processed data in a flat structure
    flat_examples = {
        'input_ids_0': tokenized_0['input_ids'],
        'attention_mask_0': tokenized_0['attention_mask'],
        'pixel_values_0': list(pixel_values_0),
        'labels_0': labels_0,
        'input_ids_1': tokenized_1['input_ids'],
        'attention_mask_1': tokenized_1['attention_mask'],
        'pixel_values_1': list(pixel_values_1),
        'labels_1': labels_1
    }

    return flat_examples


def create_labels(tokenized, tokenizer):
    """Create labels for the tokenized input based on chosen video responses."""
    labels = []
    for tokens in tokenized:
        label = [-100] * len(tokens)  # Initialize all labels to -100
        # Since we are not padding initially, we can safely assign the last tokens
        if tokens[-1:] == tokenizer("Yes", add_special_tokens=False)['input_ids']:
            label[-1:] = tokenizer("Yes", add_special_tokens=False)['input_ids']
        elif tokens[-1:] == tokenizer("No", add_special_tokens=False)['input_ids']:
            label[-1:] = tokenizer("No", add_special_tokens=False)['input_ids']
        labels.append(label)
    return labels

# Modify the function to include saving and loading
def load_and_preprocess_data(model_name, save_dir_train, save_dir_test):
    """Load and preprocess dataset with given model tokenizer, then save to disk."""
    
    # Load the dataset from a predefined path
    dataset = load_dataset(VIDEO_PATH_PREFIX)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Preprocess and map the train dataset
    train_dataset = dataset['train'].map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        batch_size=256,
        remove_columns=dataset["train"].column_names
    )

    # Save the mapped train dataset to disk
    train_dataset.save_to_disk(save_dir_train)
    print(f"Train dataset saved to {save_dir_train}")

    # Preprocess and map the test dataset
    test_dataset = dataset['test'].map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        batch_size=256,
        remove_columns=dataset["test"].column_names
    )

    # Save the mapped test dataset to disk
    test_dataset.save_to_disk(save_dir_test)
    print(f"Test dataset saved to {save_dir_test}")

    return train_dataset, test_dataset


def load_saved_data(save_dir_train, save_dir_test):
    """Load the saved datasets from disk."""
    train_dataset = load_from_disk(save_dir_train)
    test_dataset = load_from_disk(save_dir_test)

    print(f"Loaded train dataset from {save_dir_train} with {len(train_dataset)} samples")
    print(f"Loaded test dataset from {save_dir_test} with {len(test_dataset)} samples")

    return train_dataset, test_dataset


if __name__ == "__main__":
    model_name = "OpenGVLab/InternVL2-2B"
    train_save_dir = "./saved_train_dataset"
    test_save_dir = "./saved_test_dataset"

    # Step 1: Load, preprocess, and save the datasets
    train_dataset, test_dataset = load_and_preprocess_data(model_name, train_save_dir, test_save_dir)

    # Step 2: Load the saved datasets from disk
    loaded_train_dataset, loaded_test_dataset = load_saved_data(train_save_dir, test_save_dir)

    print(f"Loaded Train dataset: {loaded_train_dataset}")
    print(f"Loaded Test dataset: {loaded_test_dataset}")

