import json
import os
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from boto3.session import Session


directory = "download_files"

downlist = []

for root, dirs, files in os.walk(directory):
    for file_name in files:
        file_path = os.path.join(root, file_name)  # 获取完整文件路径
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for d in data:
                downlist.append(d["video0_body"]["video_path"])
                downlist.append(d["video1_body"]["video_path"])


def download_file(s3, bucket, file_key, download_directory):
    # 分离文件路径和文件名
    file_path, file_name = os.path.split(file_key)

    # 创建本地目录结构
    local_path = os.path.join(download_directory, file_path)
    os.makedirs(local_path, exist_ok=True)  # 创建目录，如果目录已存在则不会报错

    # 计算下载文件的完整路径
    download_path = os.path.join(local_path, file_name)
    try:
        if not os.path.exists(download_path):
            s3.download_file(Bucket=bucket, Key=file_key, Filename=download_path)
            print(f"Downloaded {file_key} to {download_path}")
    except Exception as e:
        print(f"Error downloading {file_key}: {e}")


def download_files_from_list(aws_key, aws_secret_key, bucket, region_name, file_keys, download_directory):
    # Create S3 session
    session = Session(aws_access_key_id=aws_key, aws_secret_access_key=aws_secret_key, region_name=region_name)
    s3 = session.client("s3")

    # Create the download directory if it doesn't exist
    os.makedirs(download_directory, exist_ok=True)

    # Use ThreadPoolExecutor for concurrent downloads
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for file_key in file_keys:
            futures.append(executor.submit(download_file, s3, bucket, file_key, download_directory))

        # Wait for all downloads to complete
        for future in futures:
            future.result()

def main():
    # Example test case with hardcoded parameters for testing
    aws_key = "AKIAZAI4HFAL2ABLMM6J"
    aws_secret_key = "P6Wa+43LsJRhNeMOfrQgNDn8HU8f5lIqVZG9kvgn"
    bucket = "mjbench2.0"
    region_name = "us-east-2"
    download_directory = "../videos"
    # List of S3 keys to download
    file_keys = downlist

    # Call the function to download all specified files
    download_files_from_list(aws_key, aws_secret_key, bucket, region_name, file_keys, download_directory)

if __name__ == "__main__":
    main()
