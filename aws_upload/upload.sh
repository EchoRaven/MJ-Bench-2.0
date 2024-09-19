#!/bin/bash

# Example of how to run the Python script with parameters

nohup python aws_upload_pipeline.py \
    --aws_key "AKIAZAI4HFAL6WMTRTYV" \
    --aws_secret_key "OxE33mzKLbckGlooyMYIzhb+dfdCO/8/G1zFlTqg" \
    --bucket "mjbench2.0" \
    --region_name "us-east-2" \
    --directory "/remote_shome/snl/feilong/xiapeng/haibo/image_to_video_pipeline/videos" \
    --sourceprefix "mjbench" > output.log 2>&1 &
