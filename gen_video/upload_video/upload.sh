#!/bin/bash

# Example of how to run the Python script with parameters

python3 /home/wenhao/Project/greatxue/MJ-Bench-2.0/gen_video/upload_video/aws_upload_pipeline.py \
    --aws_key "AKIAZAI4HFAL2ABLMM6J" \
    --aws_secret_key "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" \
    --bucket "mjbench2.0" \
    --region_name "us-east-2" \
    --directory "/home/wenhao/Project/greatxue/MJ-Bench-2.0/gen_video/video_to_upload/video_diffusion" \
    --sourceprefix "VidProM" 
