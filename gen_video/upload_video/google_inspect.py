from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Step 1: 进行身份验证和授权
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # 会打开浏览器让你授权

# Step 2: 创建 GoogleDrive 实例
drive = GoogleDrive(gauth)

# Step 3: 上传文件
file_path = "/home/wenhao/Project/greatxue/MJ-Bench-2.0/gen_video/videos/text_video_diffusion/text_video_diffusion_7c0be3_20240918-104438.mp4"  # 你想上传的文件路径
file_name = "inspect"  # 上传到 Google Drive 后的文件名

file = drive.CreateFile({'title': file_name})
file.SetContentFile(file_path)
file.Upload()

print(f"File '{file_name}' uploaded successfully!")