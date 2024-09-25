# videoREADME

**Deployment:**

<<<<<<< HEAD
* x Open-Sora: `https://github.com:hpcaitech/Open-Sora.git`
* o VADER: `https://github.com/mihirp1998/VADER`
* o Text-videoDiffusion: `https://modelscope.cn/models/iic/text-tovideo-synthesis`
* x instructVideo: `https://github.com/ali-vilab/VGen/blob/main/doc/InstructVideo.md`
=======
* Open-Sora x
* VADER o
* Text-videoDiffusion o
* InstructVideo x
>>>>>>> 59c308792ce44a128ff331f2f943ff68f57eb6f1

**Execution scripts:**

```python
python3 /home/wenhao/Project/greatxue/MJ-Bench-2.0/gen_video/scripts/gen_videos.py --input_file_path /home/wenhao/Project/greatxue/MJ-Bench-2.0/gen_video/input.json --output_file_path /home/wenhao/Project/greatxue/MJ-Bench-2.0/gen_video/output.json --model_list text_video_diffusion
```

**Video Saving:**

+ input part 1 -- output 5_ -- video 5 -- CUDA 0 -- v1
+ input part 5 -- output -- video -- CUDA 4 -- v2
+ input part 2 -- output 2 -- video 2 -- CUDA 0 --v3
+ input part 3 -- output 3 --video3 -- CUDA 4 -- v4
+ input part 4 -- output 4 - video 4 -- CUDA 3 -- v5

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
