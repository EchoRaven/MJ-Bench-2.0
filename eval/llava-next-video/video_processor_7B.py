import av
import torch
import numpy as np
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

# 设置模型下载目录
cache_dir = '/remote_shome/snl/feilong/xiapeng/haibo/videoRM/LLaVA-NeXT-Video/pretrained'

# 加载模型和处理器
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf", 
    torch_dtype=torch.float16, 
    device_map="auto",
    cache_dir=cache_dir
)
processor = LlavaNextVideoProcessor.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf", 
    cache_dir=cache_dir
)

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) >= len(indices):
            break
    return np.stack(frames)

def process_video(video_path, prompt):
    # 读取视频
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.linspace(0, total_frames - 1, num=8, dtype=int)

    video = read_video_pyav(container, indices)

    # 创建对话内容
    conversation = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}, {"type": "video"}],
    }]

    # 处理对话和视频
    chat_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=chat_prompt, videos=video, return_tensors="pt")

    # 将输入移动到模型所在的设备
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # 使用无梯度推理加速模型调用
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=300, use_cache=False)
    
    response = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    response_text = response[0]
    cleaned_response = response_text.split("ASSISTANT:", 1)[-1].strip()

    return cleaned_response
