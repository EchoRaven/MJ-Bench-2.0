import os
import argparse
from boto3.session import Session
from concurrent.futures import ThreadPoolExecutor, as_completed

def upload_video_to_s3(s3_client, file_path, upload_key, bucket):
    try:
        s3_client.upload_file(Filename=file_path, Key=upload_key, Bucket=bucket)
        print(f"Uploaded {os.path.basename(file_path)} to S3 as {upload_key}")
    except Exception as e:
        print(f"Error uploading {os.path.basename(file_path)}: {e}")

def upload_videos_to_s3(aws_key, aws_secret_key, bucket, region_name, directory, prefix):
    # 创建 S3 会话
    session = Session(aws_access_key_id=aws_key, aws_secret_access_key=aws_secret_key, region_name=region_name)
    s3 = session.client("s3")

    # 使用 os.walk 递归遍历目录和子目录
    files_to_upload = []
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            # 只处理视频文件，假设视频文件扩展名为 .mp4, .avi, .mov, .mkv
            if file_name.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                file_path = os.path.join(root, file_name)  # 获取完整文件路径
                
                # 生成相对路径并构建 S3 键
                relative_path = os.path.relpath(file_path, directory)  # 获取相对路径
                upload_key = f"{prefix}/{relative_path}"  # 构建上传的 S3 键，保留目录结构
                
                # 将待上传的文件添加到列表中
                files_to_upload.append((file_path, upload_key))

    # 使用 ThreadPoolExecutor 进行并发上传
    with ThreadPoolExecutor(max_workers=10) as executor:  # 可以根据需求调整 max_workers 的值
        future_to_file = {
            executor.submit(upload_video_to_s3, s3, file_path, upload_key, bucket): file_path 
            for file_path, upload_key in files_to_upload
        }

        # 获取上传结果
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                future.result()  # 检查是否有异常
            except Exception as e:
                print(f"Error occurred during upload of {file_path}: {e}")

def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Upload video files to AWS S3.")
    parser.add_argument('--aws_key', required=True, help="AWS Access Key")
    parser.add_argument('--aws_secret_key', required=True, help="AWS Secret Access Key")
    parser.add_argument('--bucket', required=True, help="S3 Bucket Name")
    parser.add_argument('--region_name', required=True, help="AWS Region")
    parser.add_argument('--directory', required=True, help="Directory Path with Video Files")
    parser.add_argument('--sourceprefix', required=True, help="Source of Dataset")

    args = parser.parse_args()

    # 调用上传函数
    upload_videos_to_s3(args.aws_key, args.aws_secret_key, args.bucket, args.region_name, args.directory, args.sourceprefix)

if __name__ == "__main__":
    main()
