import torch
import json
from swift.llm import (
    get_model_tokenizer, get_template, inference,
    get_default_template_type
)
from swift.tuners import Swift
from swift.utils import seed_everything
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
        self.inference_type = config["inference_type"]
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
                    self.base_model, config["router_path"], "router", inference_mode=True)
            self.router.generation_config.max_new_tokens = 1024
        response, _ = inference(self.router, self.template, router_template + prompt, videos=video_paths)  # chat with image
        return response


if __name__ == "__main__":
    with open("MoE_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    moe = MJ_VIDEO(config)
    video_paths = ["../videos//safesora/8cd608c47b821009baf7cc43df12b183d6da0c8c9e7125717811fa00ad4930fa/4a4c1990b549e1221e0d663a21f2970b2628059161c82af1deb6d309cf0c9ea6.mp4", "../videos//safesora/8cd608c47b821009baf7cc43df12b183d6da0c8c9e7125717811fa00ad4930fa/351b13217fc3ac1689b3f8b17356769ab7b9d36981db92462186a784f3bc57b2.mp4"]
    prompt = "2000 Documentary film in color showing dark hallway in house and kid in its center gets ripped apart from outside showing bloody monster"
    response = moe.router_choice(video_paths, prompt)
