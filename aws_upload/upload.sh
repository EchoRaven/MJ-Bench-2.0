#!/bin/bash

# Example of how to run the Python script with parameters

python aws_upload_pipeline.py \
    --aws_key "AKIAZAI4HFAL6WMTRTYV" \
    --aws_secret_key "OxE33mzKLbckGlooyMYIzhb+dfdCO/8/G1zFlTqg" \
    --bucket "mjbench2.0" \
    --region_name "us-east-2" \
    --directory "./subset" \
    --sourceprefix "subset"
