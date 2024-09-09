# directory 
project_directory/
│
├── evaluate_videos.py          
├── data.json                   
└── videos/                     
    ├── cat_playing_ball_park_0.mp4
    ├── cat_playing_ball_park_1.mp4
    ├── dog_running_field_0.mp4
    └── dog_running_field_1.mp4
    
# json example
···
{
    "examples": [
        {
            "caption": "A cat playing with a ball in the park",
            "video0": "videos/cat_playing_ball_park_0.mp4",
            "video1": "videos/cat_playing_ball_park_1.mp4"
        },
        {
            "caption": "A dog running in the field",
            "video0": "videos/dog_running_field_0.mp4",
            "video1": "videos/dog_running_field_1.mp4"
        }
    ]
}
···

# deal with files
···
import json

def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['examples']

if __name__ == "__main__":
    # 读取输入数据
    examples = load_data("data.json")
    
    # 遍历每个例子并评估视频
    for example in examples:
        caption = example["caption"]
        video0 = example["video0"]
        video1 = example["video1"]
        
        # 调用你的评估函数
        scores = evaluate_videos(caption, video0, video1)
        print(scores)
···
