import argparse
from boto3.session import Session
import os

def download_single_file_from_s3(aws_key, aws_secret_key, bucket, region_name, file_key, download_directory):
    # Create S3 session
    session = Session(aws_access_key_id=aws_key, aws_secret_access_key=aws_secret_key, region_name=region_name)
    s3 = session.client("s3")

    # Create the download directory if it doesn't exist
    os.makedirs(download_directory, exist_ok=True)

    # Construct the full path to save the file
    file_name = os.path.basename(file_key)  # Get just the filename from the key
    download_path = os.path.join(download_directory, file_name)

    # Download the specified file from S3
    try:
        s3.download_file(Bucket=bucket, Key=file_key, Filename=download_path)
        print(f"Downloaded {file_key} to {download_path}")
    except Exception as e:
        print(f"Error downloading {file_key}: {e}")

def main():
    # Example test case with hardcoded parameters for testing
    aws_key = "AKIAZAI4HFAL6WMTRTYV"
    aws_secret_key = "OxE33mzKLbckGlooyMYIzhb+dfdCO/8/G1zFlTqg"
    bucket = "mjbench2.0"
    region_name = "us-east-2"
    file_key = "mjbench_alignment_image0_0.mp4"  # The file to download from S3 (S3 key)
    download_directory = "./downloaded_files"  # Directory to save the downloaded file

    # Call the function to download the specific file
    download_single_file_from_s3(aws_key, aws_secret_key, bucket, region_name, file_key, download_directory)

if __name__ == "__main__":
    main()
