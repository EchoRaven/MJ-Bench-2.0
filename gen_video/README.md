# README

**Deployment:**

* Open-Sora x
* VADER o
* Text-videoDiffusion o
* nstructVideo x

**Execution scripts:**

```python
python3 /home/wenhao/Project/greatxue/MJ-Bench-2.0/gen_video/scripts/gen_videos.py --input_file_path /home/wenhao/Project/greatxue/MJ-Bench-2.0/gen_video/input.json --output_file_path /home/wenhao/Project/greatxue/MJ-Bench-2.0/gen_video/output.json --model_list text_video_diffusion
```

**File Structure:**

```plaintext
gen_video/
├── api_code/
│   ├── gen_instruct_video.py
│   ├── gen_opensora.py
│   ├── gen_text_video_diffussion.py
│   ├── gen_vader.py
├── scripts/
│   ├── clone_repos.sh
│   ├── gen_videos.sh
├── videos/
│   ├── instructvideo/
│   │   ├── video_instructvideo.mp4
│   ├── opensora/
│   │   ├── video_opensora.mp4
│   ├── text_video_diffusion/
│   │   ├── video_text_video_diffusion.mp4
│   ├── vader/
│   │   ├── video_vader.mp4
├── input.json
├── output.json
├── README.md
```
