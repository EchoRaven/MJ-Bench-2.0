import json

config_data = {
    "general_settings": {
        "model_name": "OpenGVLab/InternVL2-26B",
        "task_type": "text-generation"
    },
    "task_settings": {
        "prompt": "A cat playing with the fluffy ball",
        "output_path": "generated_video.mp4"
    },
    "optional_settings": {
        "trust_remote_code": True
    }
}



from sqlalchemy import true
from transformers import pipeline
import os

with open('config.json', 'r') as json_file:
    config = json.load(json_file)


video_gen = pipeline(
    config_data['general_settings']['task_type'],
    model=config_data['general_settings']['model_name'],
    trust_remote_code=config_data['optional_settings']['trust_remote_code']
)


generated_video = video_gen(config_data['task_settings']['prompt'])


output_path = config_data['task_settings']['output_path']
with open(output_path, "wb") as f:
    f.write(generated_video["video"])


if os.path.exists(output_path):
    print(f"Video successfully generated and saved as {output_path}")
else:
    print("Error in video generation or saving process.")



#Generate pipeline
# Step1: Define the prompt
prompt = "A cat playing with the fluffy ball"

# Step2: Load the pre-trained text-to-video model (replace with the best available)
model_name = "OpenGVLab/InternVL2-26B"
# Use a pipeline as a high-level helper
video_gen = pipeline("text-generation", model=model_name, trust_remote_code=True)

# Step3: Generate the video from the prompt
generated_video = video_gen(prompt)

# Step4: Save the generated video to a file
output_path = "generated_video.mp4"
with open(output_path, "wb") as f:
    f.write(generated_video["video"])

# Optional: Confirm the video is saved successfully
if os.path.exists(output_path):
    print(f"Video successfully generated and saved as {output_path}")
else:
    print("Error in video generation or saving process.")
