from diffusers import DiffusionPipeline
from PIL import Image
import torch

# 设置模型要保存的文件夹路径
custom_cache_dir = "/remote_shome/snl/feilong/xiapeng/haibo/videoRM/Stable_Video_Diffusion"

# 从预训练模型加载，同时指定cache_dir参数
pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", cache_dir=custom_cache_dir)
pipeline = pipeline.to("cuda")  # 使用GPU加速

# 加载输入图像
image_path = "input_image.jpg"  # 替换为你要加载的图像路径
input_image = Image.open(image_path).convert("RGB")

# 获取输入图像的尺寸
input_image_size = input_image.size

# 使用pipeline生成视频
with torch.no_grad():
    video_frames = pipeline(input_image, num_frames=25).frames[0]  # 生成40帧视频

# 调整每个生成的帧的尺寸与输入图像一致
resized_frames = [frame.resize(input_image_size) for frame in video_frames]

output_video_path = "output_video.gif"  # 保存路径

# 保存为 GIF
resized_frames[0].save(output_video_path, format='GIF',
                       save_all=True, append_images=resized_frames[1:],
                       duration=100, loop=0)

print(f"GIF文件已保存为: {output_video_path}")
