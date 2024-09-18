'''
import argparse
import os

########################################### ONLY FOR TEST ##########################################
def generate_video(prompt, output_path):
    print(f"Generating video for prompt: {prompt} using OpenSora...")
    #TODO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate video using OpenSora')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for video generation')
    parser.add_argument('--output', type=str, required=True, help='Output video file path')
    
    args = parser.parse_args()
    
    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    generate_video(args.prompt, args.output)
'''

import argparse
import os
import random
import string

########################################### ONLY FOR TEST ##########################################
def generate_random_mp4_file(output_path):
    random_filename = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10)) + ".mp4"
    full_path = os.path.join(output_path)
    with open(full_path, 'wb') as f:
        f.write(os.urandom(1024)) 

    print(f"Random MP4 file generated at: {full_path}")
    return full_path

def generate_video(prompt, output_path):
    print(f"Generating video for prompt: {prompt}...")
    generate_random_mp4_file(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate video using OpenSora')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for video generation')
    parser.add_argument('--output', type=str, required=True, help='Output video file path')

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generate_video(args.prompt, args.output)