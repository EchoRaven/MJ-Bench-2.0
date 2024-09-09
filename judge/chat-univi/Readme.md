# Project Directory Overview

This project directory is designed to evaluate the alignment between a text description (caption) and two videos. Below is the structure of the directory and an example of how the videos and data files are organized.

## Directory Structure

```
project_directory/
│
├── evaluate_videos.py        # Main script for evaluating the videos
├── data.json                 # JSON file containing the video descriptions and paths
└── videos/                   # Folder containing the videos to be evaluated
    ├── cat_playing_ball_park_0.mp4
    ├── cat_playing_ball_park_1.mp4
    ├── dog_running_field_0.mp4
    └── dog_running_field_1.mp4
```

## Example JSON Structure

The `data.json` file contains a list of examples. Each example consists of:
- `caption`: A text description of the video content.
- `video0` and `video1`: Paths to the two videos being compared against the caption.

### JSON Format:

```json
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
```

## Handling the Files

The following Python script demonstrates how to read the `data.json` file and evaluate each pair of videos based on the provided caption. You can integrate your evaluation function to process the videos and output the scores.

### Python Script:

```python
import json

def load_data(json_file):
    """
    Loads data from a JSON file and returns the list of examples.
    
    Parameters:
        json_file (str): Path to the JSON file containing the examples.
        
    Returns:
        list: List of examples where each example contains a caption and two video paths.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['examples']

def evaluate_videos(caption, video0, video1):
    """
    Placeholder function to evaluate videos based on the caption.
    
    Parameters:
        caption (str): The text description of the video content.
        video0 (str): Path to the first video.
        video1 (str): Path to the second video.
        
    Returns:
        dict: Evaluation scores for the two videos.
    """
    # Your evaluation logic should go here
    return {"score_video0": 4.5, "score_video1": 3.7}

if __name__ == "__main__":
    # Load input data from the JSON file
    examples = load_data("data.json")
    
    # Iterate over each example and evaluate the videos
    for example in examples:
        caption = example["caption"]
        video0 = example["video0"]
        video1 = example["video1"]
        
        # Call your evaluation function
        scores = evaluate_videos(caption, video0, video1)
        print(f"Scores for caption '{caption}':", scores)
```

## Instructions:
1. Place your videos in the `videos/` folder.
2. Update the `data.json` file with captions and paths to your videos.
3. Customize the `evaluate_videos` function with your own logic to score the alignment between the caption and videos.
4. Run the script `evaluate_videos.py` to evaluate and print the scores for each example.

This setup allows you to easily add new video pairs and captions to the `data.json` file and automatically process them through your evaluation system.
