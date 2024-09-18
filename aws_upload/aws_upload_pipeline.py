import os
import argparse
from boto3.session import Session

def upload_videos_to_s3(aws_key, aws_secret_key, bucket, region_name, directory):
    # 创建 S3 会话
    session = Session(aws_access_key_id=aws_key, aws_secret_access_key=aws_secret_key, region_name=region_name)
    s3 = session.client("s3")

    # 遍历指定目录，上传所有视频文件
    for file_name in os.listdir(directory):
        # 只处理视频文件，假设视频文件扩展名为 .mp4, .avi 等
        if file_name.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            file_path = os.path.join(directory, file_name)  # 获取完整文件路径
            upload_key = f"mjbench_{file_name}"  # 构建上传的 S3 键

            # 上传文件到 S3
            try:
                s3.upload_file(Filename=file_path, Key=upload_key, Bucket=bucket)
                print(f"Uploaded {file_name} to S3 as {upload_key}")
            except Exception as e:
                print(f"Error uploading {file_name}: {e}")

def main():
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Upload video files to AWS S3.")
    parser.add_argument('--aws_key', required=True, help="AWS Access Key")
    parser.add_argument('--aws_secret_key', required=True, help="AWS Secret Access Key")
    parser.add_argument('--bucket', required=True, help="S3 Bucket Name")
    parser.add_argument('--region_name', required=True, help="AWS Region")
    parser.add_argument('--directory', required=True, help="Directory Path with Video Files")

    args = parser.parse_args()

    # 调用上传函数
    upload_videos_to_s3(args.aws_key, args.aws_secret_key, args.bucket, args.region_name, args.directory)

if __name__ == "__main__":
    main()
