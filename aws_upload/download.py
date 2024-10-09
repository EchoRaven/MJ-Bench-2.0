import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from boto3.session import Session

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

def download_files_with_prefix_from_s3(aws_key, aws_secret_key, bucket, region_name, prefix, download_directory):
    # Create S3 session
    session = Session(aws_access_key_id=aws_key, aws_secret_access_key=aws_secret_key, region_name=region_name)
    s3 = session.client("s3")

    # Create the download directory if it doesn't exist
    os.makedirs(download_directory, exist_ok=True)

    # List all the objects under the given prefix
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' not in response:
            print(f"No files found with prefix: {prefix}")
            return

        # Use ThreadPoolExecutor for concurrent downloads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for obj in response['Contents']:
                file_key = obj['Key']
                futures.append(executor.submit(download_file, s3, bucket, file_key, download_directory))

            # Wait for all downloads to complete
            for future in futures:
                future.result()

    except Exception as e:
        print(f"Error listing objects with prefix {prefix}: {e}")

def main():
    # Example test case with hardcoded parameters for testing
    aws_key = "AKIAZAI4HFAL2ABLMM6J"
    aws_secret_key = "P6Wa+43LsJRhNeMOfrQgNDn8HU8f5lIqVZG9kvgn"
    bucket = "mjbench2.0"
    region_name = "us-east-2"
    prefix = "test"
    download_directory = "../test"

    # Call the function to download all files with the specified prefix
    download_files_with_prefix_from_s3(aws_key, aws_secret_key, bucket, region_name, prefix, download_directory)

if __name__ == "__main__":
    main()
