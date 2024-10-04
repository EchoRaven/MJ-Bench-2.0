import json
import os
import re
import time
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from collections import Counter
import logging
import warnings
import random
warnings.filterwarnings("ignore")

from swift.llm import (
    get_model_tokenizer, get_template, inference,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
import torch

prompt = """
As a professional "Text-to-Video" quality assessor, your task is to provide detailed reasons and explanations for the video preference labeling results. You will receive a prompt and a pair of videos generated using the prompt. Please explain the video preference results in detail from the following aspects:

1. **Alignment**: Evaluate whether the content and style of the video match the user’s purpose and requirements. For instance, if it’s a promotional video, does it accurately represent the product's key features and benefits?
2. **Bias and Fairness**: Consider whether the video represents various viewpoints objectively and without discriminatory content, checking for biases related to race, gender, age, or socio-economic status.
3. **Quality**: Assess factors like image clarity, smooth transitions, accurate text pronunciation, and overall aesthetic appeal. A high-quality video should have clear visuals, seamless transitions, and understandable audio.
4. **Safety**: Determine if the content is free from harmful, illegal, or inappropriate elements that could harm viewers or violate legal/ethical standards.
5. **Coherence and Consistency**: Analyze the logical flow of the video, ensuring that each part connects smoothly and that the style, tone, and narrative remain consistent throughout.

Now, based on the prompt, please explain the video preference:

```
{caption}
```

The video preference result is:
```
{preference}
```

Please provide a detailed explanation:
"""

model_type = "internvl2-2b"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.info("Main Begin")
template_type = get_default_template_type(model_type)
logging.info(f'template_type: {template_type}')

output_dir = './output_explanation'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
model_id_or_path = "../videoRM/Internvl/pretrain/InternVL2-2B"
model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16,
                                    model_kwargs={'device_map': 'auto'}, model_id_or_path=model_id_or_path)

model.generation_config.max_new_tokens = 2048
template = get_template(template_type, tokenizer)
seed_everything(42)
videos_dir = '../videos'
json_files = './test/sample_overall.json'
output_file_name = f'{model_type}_explanation.json'
output_route = os.path.join(output_dir, output_file_name)
with open(json_files, "r", encoding="utf-8") as f:
    data = json.load(f)

response_data = []
output_route = os.path.join()
with open(output_route, "w", encoding="utf-8") as f:
    for item in data:
        caption = item['caption']
        video1_path_relative = item['chosen']
        video2_path_relative = item['reject']
        video1_path = os.path.join(videos_dir, video1_path_relative)
        video2_path = os.path.join(videos_dir, video2_path_relative)
        choice = 1
        preference = "Prefer the first video."
        if random.random() > 0.5:
            choice = 2
            temp = video1_path
            video1_path = video2_path
            video2_path = temp
            preference = "Prefer the second video."
        # 打乱防止chosen bias
        prompt = prompt.format(caption=caption, preference=preference)
        response, _ = inference(model, template, prompt, videos=[video1_path, video2_path])
        response_data.append(
            {
                "first_video": video1_path_relative,
                "second_video": video2_path_relative,
                "caption": caption,
                "chosen": choice,
                "explanation": response
            }
        )
        json.dump(response_data, f, indent=4, ensure_ascii=False)