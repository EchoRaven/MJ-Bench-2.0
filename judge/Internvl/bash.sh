#!/bin/bash

python evaluate_videos.py \
    --json_file_path "/remote_shome/snl/feilong/xiapeng/haibo/videoRM/dataset/SafeSora/config-train.json" \
    --videos_dir "/remote_shome/snl/feilong/xiapeng/haibo/videoRM/dataset/SafeSora" \
    --model_path "OpenGVLab/InternVL2-8B" \
    --cache_dir "/remote_shome/snl/feilong/xiapeng/haibo/videoRM/Internvl/pretrained" \
    --alignment_results_file "alignment_result.json" \
    --safety_results_file "safety_result.json" \
    --bias_results_file "bias_result.json" \
    --quality_results_file "quality_result.json" \
    --cc_results_file "cc_result.json"
