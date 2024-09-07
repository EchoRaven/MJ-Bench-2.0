# call_video_moderator.py

import sys
import os
from video_moderator import VideoModerator

# Make sure the `utility.models.VidLLM` module is in your PYTHONPATH
sys.path.append('/home/llm/utility/models/VidLLM')

def main():
    # Initialize the VideoModerator
    model_id = 'InternVL2-8B'  # Example model_id; replace with your actual model_id
    device = 'cuda'  # or 'cpu'
    ckpt_dir = None  # or specify a checkpoint directory if needed
    
    moderator = VideoModerator(model_id=model_id, device=device, ckpt_dir=ckpt_dir)
    
    # Path to your video files and description
    video_path1 = 'path/to/your/first_video.mp4'
    video_path2 = 'path/to/your/second_video.mp4'
    description = 'Describe the content of the video.'
    
    # Fixed question
    question = 'Rate the alignment between the description and the video on a scale of 1 to 5.'

    # Generate scores for both videos
    score1 = moderator.evaluate(description, video_path1, question)
    score2 = moderator.evaluate(description, video_path2, question)
    
    # Print scores
    print(f"Score for Video 1: {score1}")
    print(f"Score for Video 2: {score2}")

    # Compare scores and determine which video is better
    if score1 > score2:
        print("Video 1 is better aligned with the description.")
    elif score1 < score2:
        print("Video 2 is better aligned with the description.")
    else:
        print("Both videos are equally aligned with the description.")

if __name__ == "__main__":
    main()
