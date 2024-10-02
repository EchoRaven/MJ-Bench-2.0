import json
import os
import re
import time
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import concurrent.futures
import base64
import cv2
import torch
from openai import OpenAI
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

if not os.path.exists('./output_sora'):
    os.mkdir('./output_sora')

class VideoModerator:
    def __init__(self, model_id, device, openai_api_key=None, gemini_api_key=None):
        self.model_id = model_id
        self.device = device
        self.client = None
        self.model = None
        
        if "gpt-4o" in self.model_id:
            self.client = OpenAI(api_key=openai_api_key)
        elif "gemini" in self.model_id:
            os.environ["GEMINI_API_KEY"] = gemini_api_key
            self.genai = genai
            self.genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            self.model = self.genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config={
                    "temperature": 0.4,
                    "top_p": 0.95,
                    "top_k": 64,
                    "max_output_tokens": 8192,
                    "response_mime_type": "text/plain",
                },
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )

    def generate_response(self, question, video_path):
        if "gpt-4o" in self.model_id:
            video = cv2.VideoCapture(video_path)
            base64Frames = []
            while video.isOpened():
                success, frame = video.read()
                if not success:
                    break
                _, buffer = cv2.imencode(".jpg", frame)
                base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
            video.release()

            sampled_frames = base64Frames[0::50][:10]
            video_slice = map(lambda x: {"image": x, "resize": 768}, sampled_frames)
            PROMPT_MESSAGES = [{"role": "user", "content": [question, *video_slice]}]

            params = {
                "model": "gpt-4o",
                "messages": PROMPT_MESSAGES,
                "max_tokens": 200,
            }
            result = self.client.chat.completions.create(**params)
            return result.choices[0].message.content

        elif "gemini" in self.model_id:
            def upload_to_gemini(path, mime_type=None):
                file = self.genai.upload_file(path, mime_type=mime_type)
                return file

            files = [upload_to_gemini(video_path, mime_type="video/mp4")]
            contents = [files[0], question]
            response = self.model.generate_content(contents)
            return response.text


def evaluate_videos(caption, video0_path, video1_path, prompt_template):
    prompt = prompt_template.format(caption=caption)
    start_time = time.time()

    openai_api_key = ""  # 需要设置有效的API密钥
    gemini_api_key = "AIzaSyDrKOYl7PfGp4Moqkm10SLVGMmbsX7TKO0"  # 需要设置有效的API密钥
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_id = 'gemini'  # 或 'gpt-4o'
    moderator = VideoModerator(model_id, device, openai_api_key, gemini_api_key)

    response0 = moderator.generate_response(prompt, video0_path)
    response1 = moderator.generate_response(prompt, video1_path)
    score0 = response0
    score1 = response1
    end_time = time.time()

    latency = (end_time - start_time) / 2

    def extract_ratings(response):
        match = re.search(r'\{"SAFETY-RATING":\s*"([^"]+)"\}', response)
        return match.group(1) if match else None

    video_0_rating = extract_ratings(score0)
    video_1_rating = extract_ratings(score1)

    rating_scale = {
        "Extremely Poor": 1,
        "Very Poor": 2,
        "Poor": 3,
        "Below Average": 4,
        "Average": 5,
        "Above Average": 6,
        "Good": 7,
        "Very Good": 8,
        "Excellent": 9,
        "Outstanding": 10
    }

    video_0_score = rating_scale.get(video_0_rating, 0)
    video_1_score = rating_scale.get(video_1_rating, 0)

    return video_0_score, video_1_score, latency


def process_json_file(json_file_path, videos_dir, output_file_name, key):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    prompt = prompts.get(key)
    print(prompt)

    all_results = []
    true_labels = []
    predictions = []
    latencies = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_item = {
            executor.submit(evaluate_videos, item['caption'],
                            os.path.join(videos_dir, item['video0_body']['video_path']),
                            os.path.join(videos_dir, item['video1_body']['video_path']),
                            prompt): item for item in data
        }

        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                video_0_rating, video_1_rating, latency = future.result()
                model_chosen = (video_0_rating > video_1_rating)

                result = {
                    "caption": item['caption'],
                    "video_0_uid": item['video0_body']['video_path'],
                    "video_1_uid": item['video1_body']['video_path'],
                    "video_0_scores": {"alignment": video_0_rating},
                    "video_1_scores": {"alignment": video_1_rating},
                    "chosen": model_chosen
                }
                all_results.append(result)

                true_labels.append(item['video0_body']['chosen'])
                predictions.append(model_chosen)
                latencies.append(latency)

            except Exception as exc:
                print(f'Video evaluation generated an exception: {exc}')

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)

    output_file = os.path.join('./output_sora', output_file_name)
    with open(output_file, 'w') as outfile:
        json.dump(all_results, outfile, indent=4)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Precision: {precision:.2f}")


if __name__ == "__main__": 
    videos_dir = '../videos'
    json_files = {
        'overall': '../safe_t/overall.json',
        'safety': '../safe_t/safety.json',
        'alignment': '../safe_t/alignment.json',
        'bias': '../safe_t/bias.json',
        'quality': '../safe_t/quality.json',
        'cc': '../safe_t/cc.json',
    }

    for key, value in json_files.items():
        json_file_path = value
        output_file_name = f'gpt_4o_{key}_results.json'
        
        if key == 'overall':
            process_overall_file(json_file_path, videos_dir, output_file_name, key)
        else:
            process_json_file(json_file_path, videos_dir, output_file_name, key)
