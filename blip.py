import os
import cv2
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

# 设置设备
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# 加载模型和预处理器
model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)

# 视频采样成图片
def sample_video(video_path, output_dir, sample_rate=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    img_count = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_rate == 0:  # 每采样率获取一帧
            img_path = os.path.join(output_dir, f"frame_{img_count}.jpg")
            cv2.imwrite(img_path, frame)
            img_count += 1
        frame_count += 1

    cap.release()

# 计算图像与文本的相似度（ITM 和 ITC）
def compute_similarity(image_path, caption):
    # 加载并预处理图像和文本
    raw_image = Image.open(image_path).convert("RGB")
    img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    txt = text_processors["eval"](caption)

    # 计算 ITM 分数
    itm_output = model({"image": img, "text_input": txt}, match_head="itm")
    itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
    itm_score = itm_scores[:, 1].item()

    # 计算 ITC 分数
    itc_score = model({"image": img, "text_input": txt}, match_head="itc")

    return itm_score, itc_score.item()

# 计算加权分数
def compute_weighted_score(image_paths, caption):
    total_itm_score = 0
    total_itc_score = 0
    total_weight = 0

    for i, img_path in enumerate(image_paths):
        itm_score, itc_score = compute_similarity(img_path, caption)
        weight = 1  # 你可以根据需要调整权重，比如根据帧的位置
        total_itm_score += itm_score * weight
        total_itc_score += itc_score * weight
        total_weight += weight

    avg_itm_score = total_itm_score / total_weight
    avg_itc_score = total_itc_score / total_weight

    return avg_itm_score, avg_itc_score

# 主函数，比较两个视频的生成效果
def compare_videos(video1_path, video2_path, caption, sample_rate=30):
    # 采样视频1和视频2
    sample_video(video1_path, "./video1_frames", sample_rate)
    sample_video(video2_path, "./video2_frames", sample_rate)

    # 获取采样后的视频帧路径
    video1_images = [os.path.join("./video1_frames", f) for f in sorted(os.listdir("./video1_frames")) if f.endswith(".jpg")]
    video2_images = [os.path.join("./video2_frames", f) for f in sorted(os.listdir("./video2_frames")) if f.endswith(".jpg")]

    # 计算两个视频的加权相似度分数
    video1_itm_score, video1_itc_score = compute_weighted_score(video1_images, caption)
    video2_itm_score, video2_itc_score = compute_weighted_score(video2_images, caption)

    # 输出结果
    print(f"Video 1 ITM score: {video1_itm_score:.4f}, ITC score: {video1_itc_score:.4f}")
    print(f"Video 2 ITM score: {video2_itm_score:.4f}, ITC score: {video2_itc_score:.4f}")

    if video1_itm_score > video2_itm_score:
        print("Video 1 更加贴切")
    else:
        print("Video 2 更加贴切")

# 测试函数
if __name__ == "__main__":
    video1_path = "path_to_video1.mp4"  # 替换为你的视频1路径
    video2_path = "path_to_video2.mp4"  # 替换为你的视频2路径
    caption = "merlion in Singapore"  # 替换为你的描述文本

    # 比较两个视频
    compare_videos(video1_path, video2_path, caption, sample_rate=30)
