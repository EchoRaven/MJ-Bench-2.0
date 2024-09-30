import torch
import json
from swift.llm import (
    get_model_tokenizer, get_template, inference,
    get_default_template_type
)
from swift.tuners import Swift
from swift.utils import seed_everything
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import re
import json

def convert_to_json_format(input_string: str) -> dict:
    # 使用正则表达式为键添加引号
    json_compatible_string = re.sub(r"(\w+):", r'"\1":', input_string)

    # 将单引号替换为双引号
    json_string = json_compatible_string.replace("'", '"')

    # 将其转换为JSON对象并返回
    return json.loads(json_string)

class MJ_VIDEO:
    def __init__(self, config):
        self.config = config
        self.dtype = torch.bfloat16 if config["dtype"] == "bfloat16" else torch.float32
        logging.info("Loading base model ...")
        self.base_model, self.tokenizer = get_model_tokenizer(config["model_type"], self.dtype,
                            model_kwargs={'device_map': 'auto'}, model_id_or_path=config["model_id_or_path"])
        template_type = get_default_template_type(config["model_type"])
        self.template = get_template(template_type, self.tokenizer)
        seed_everything(42)

        # load prompt list
        with open(config["prompt_list"], "r", encoding="utf-8") as f:
            self.prompt_list = json.load(f)

        # define MoE
        if config["inference_type"] == "high_speed":
            # define router
            logging.info("Loading router ...")
            self.router = Swift.from_pretrained(
                      self.base_model, config["router_path"], "router", inference_mode=True)
            self.router.generation_config.max_new_tokens = 1024
            # define experts
            self.expert_group = {}
            for key in config["experts"].keys():
                logging.info(f"Loading {key} expert ...")
                self.expert_group[key] = Swift.from_pretrained(
                      self.base_model, config["experts"][key], key, inference_mode=True)
                self.expert_group[key].generation_config.max_new_tokens = 1024

    def router_choice(self, video_paths, prompt):
        router_template = self.prompt_list["router"]
        if self.config["inference_type"] == "low_cost":
            self.router = Swift.from_pretrained(
                    self.base_model, self.config["router_path"], "router", inference_mode=True)
            self.router.generation_config.max_new_tokens = 1024
        response, _ = inference(self.router, self.template, router_template + prompt, videos=video_paths)
        if self.config["inference_type"] == "low_cost":
            del self.router
        return response
    
    def activate_expert(self, force_keys, router_response):
        if len(force_keys) > 0:
            return force_keys
        experts = []
        response_result = convert_to_json_format(router_response)
        for key in response_result.keys():
            if response_result[key] == "yes":
               experts.append(key)
        return experts
    
    def process_expert(self, expert, video_paths, prompt):

        if self.config["inference_type"] == "low_cost":
            self.expert_group[expert] = Swift.from_pretrained(
                self.base_model, self.config["experts"][expert], expert, inference_mode=True)
            self.expert_group[expert].generation_config.max_new_tokens = 1024

        expert_template = self.prompt_list[expert]
        response, _ = inference(self.expert_group[expert], self.template, expert_template + prompt, videos=video_paths)

        if self.config["inference_type"] == "low_cost":
            del self.expert_group[expert]

        response_json = convert_to_json_format(response)
        return response_json, expert
    
    def experts_judge(self, experts, video_paths, prompt):
        def process_expert_concurrently(expert):
            return self.process_expert(expert, video_paths, prompt)
        result = {}
        if self.config["inference_type"] == "low_cost":
            for expert in experts:
                result_json, _ = process_expert_concurrently(expert, video_paths, prompt)
                result[expert] = result_json
        else:
            with ThreadPoolExecutor() as executor:
                future_to_expert = {executor.submit(process_expert_concurrently, expert): expert for expert in experts}
                for future in as_completed(future_to_expert):
                    expert = future_to_expert[future]
                    try:
                        result_json, _ = future.result()
                        result[expert] = result_json
                    except Exception as exc:
                        print(f'{expert} generated an exception: {exc}')
            return result
        
    def judge(self, video_paths, prompt, force_keys=[]):
        router_response = self.router_choice(video_paths, prompt)
        experts = self.activate_expert(force_keys, router_response)
        experts_response = self.experts_judge(experts, video_paths, prompt)
        return experts_response
    
    def inference(self, video_paths, prompt, force_keys=[]):
        judge_result = self.judge(video_paths, prompt, force_keys)
        response = ""
        score_1 = 0
        score_2 = 0
        for expert in judge_result.keys():
            labels = judge_result[expert]
            response += f"From the perspective of {expert}, "
            first_better = []
            second_better = []
            for label in labels.keys():
                mark = labels[label]
                if mark == "first":
                    first_better.append(label)
                elif mark == "second":
                    second_better.append(label)
            if len(first_better) > 0:
                response += "video 1 performs better in terms of "
                for label in first_better:
                    response += f"{label}, "
            if len(second_better) > 0:
                if len(first_better) != 0:
                    response += "while video 2 performs better in terms of "
                else:
                    response += "video 2 performs better in terms of "
                for label in second_better:
                    response += f"{label}, "
                response = response[:-2] + ". "
            score_1 += len(first_better)
            score_2 += len(second_better)
        response += "As a result, "
        if score_1 > score_2:
            response += "video 1 is better."
            return response, "video 1"
        elif score_1 < score_2:
            response += "video 2 is better."
            return response, "video 2"
        else:
            response += "they are the same."
            return response, "same"
        

if __name__ == "__main__":
    with open("MoE_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    moe = MJ_VIDEO(config)
    video_paths = ["../videos//safesora/8cd608c47b821009baf7cc43df12b183d6da0c8c9e7125717811fa00ad4930fa/4a4c1990b549e1221e0d663a21f2970b2628059161c82af1deb6d309cf0c9ea6.mp4", "../videos//safesora/8cd608c47b821009baf7cc43df12b183d6da0c8c9e7125717811fa00ad4930fa/351b13217fc3ac1689b3f8b17356769ab7b9d36981db92462186a784f3bc57b2.mp4"]
    prompt = "2000 Documentary film in color showing dark hallway in house and kid in its center gets ripped apart from outside showing bloody monster"
    response, result = moe.inference(video_paths, prompt)
    print(response)
    print(result)
