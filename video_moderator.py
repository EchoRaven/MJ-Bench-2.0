import torch
import transformers
import json
import sys, os
from utility.models.VidLLM.videollama import *
from utility.models.VidLLM.videollama import load_pretrained_model as load_pretrained_model_videollama
from utility.models.VidLLM.videollama import conv_templates as conv_templates_videollama
from utility.models.VidLLM.internvl2 import *
from utility.models.VidLLM.chatunivi import *
from utility.models.VidLLM.chatunivi import _get_rawvideo_dec
from utility.models.VidLLM.chatunivi import load_pretrained_model as load_pretrained_model_chatunivi
from utility.models.VidLLM.chatunivi import conv_templates as conv_templates_chatunivi
from utility.models.VLLM.frame_based import *

from tqdm import tqdm
# from autoshot import load_autoshot
import time


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {'InternVL2-8B': 32, 'InternVL2-26B': 48,
                  'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

class VideoModerator:
    def __init__(self, model_id, device, ckpt_dir=None):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.model_path = None
        self.model_name = None
        overwrite_path = ckpt_dir
    
        if "VideoLLaMA" in self.model_id:
            
            # 1. Initialize the model.
            # model_path = '/scratch/czr/hf_models/VideoLLaMA2-7B'
            model_path = '/scratch/czr/hf_models/VideoLLaMA2-7B-16F'
            if overwrite_path is not None:
                model_path = overwrite_path
            # Base model inference (only need to replace model_path)
            model_name = get_model_name_from_path(model_path)
            self.tokenizer, self.model, self.processor, self.context_len = load_pretrained_model_videollama(model_path, None, model_name)
            self.model = self.model.to(device)
            self.conv_mode = 'llama_2'

        elif "InternVL2" in self.model_id:
            
            model_path = f'/scratch/czr/hf_models/{self.model_id}'
            # If you have an 80G A100 GPU, you can put the entire model on a single GPU.
            if overwrite_path is not None:
                model_path = overwrite_path
            try:
                device_map = split_model(self.model_id)
                print(device_map)

                self.model = AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    # load_in_8bit=True,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map=device_map,
                    ).eval()#.to(device)
            
            except Exception as e:
                self.model = AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    # load_in_8bit=True,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    ).eval().to(device)

            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            # set the max number of tiles in `max_num`
            # policy_start_token = self.tokenizer('<box>')
            # policy_end_token = self.tokenizer('</box>')
            # print(policy_start_token, policy_end_token)
            # input()
            self.generation_config = dict(
                num_beams=1,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.5,
            )
        
        elif "Chat-UniVi" in self.model_id:
            import os
            model_path = "/scratch/czr/hf_models/Chat-UniVi-13B"
            if overwrite_path is not None:
                model_path = overwrite_path
            # The number of visual tokens varies with the length of the video. "max_frames" is the maximum number of frames.
            # When the video is long, we will uniformly downsample the video to meet the frames when equal to the "max_frames".
            self.max_frames = 100

            # The number of frames retained per second in the video.
            self.video_framerate = 1

            # Sampling Parameter
            self.conv_mode = "guardrail"
            self.temperature = 0.2
            self.top_p = None
            self.num_beams = 1

            disable_torch_init()
            model_path = os.path.expanduser(model_path)
            model_name = "ChatUniVi"
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model_chatunivi(model_path, None, model_name)

            mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
            mm_use_im_patch_token = getattr(self.model.config, "mm_use_im_patch_token", True)
            if mm_use_im_patch_token:
                tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            if mm_use_im_start_end:
                tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.model.resize_token_embeddings(len(self.tokenizer))

            vision_tower = self.model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            self.image_processor = vision_tower.image_processor

            # if self.model.config.config["use_cluster"]:
            #     for n, m in self.model.named_modules():
            #         m = m.to(dtype=torch.bfloat16)

        elif "MiniCPM" in model_id:
            # Load the model and tokenizer
            model_path = "/scratch/czr/hf_models/MiniCPM-Llama3-V-2_5"
            if overwrite_path is not None:
                model_path = overwrite_path
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)#, torch_dtype=torch.float16)
            self.model = self.model.to(device='cuda')
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model.eval()
        elif "llava-v1.6-mistral" in model_id:
            if overwrite_path is not None:
                model_path = overwrite_path
            self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
            self.model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16) 
            self.model.to("cuda")
        elif "gpt-4o" in model_id:
            from openai import OpenAI
            self.client = OpenAI(api_key="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")
        elif "gemini" in model_id:
            import os
            os.environ["GEMINI_API_KEY"] = "AIzaSyBC0GUWf455UBar3HHIRw92mNA3-FgJGAM"
            import time
            import google.generativeai as genai
            from google.generativeai.types import HarmCategory, HarmBlockThreshold
            self.genai = genai

            self.genai.configure(api_key=os.environ["GEMINI_API_KEY"])

            generation_config = {
            "temperature": 0.4,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
            }

            self.model = self.genai.GenerativeModel(
            model_name = "gemini-1.5-flash",
            generation_config=generation_config,
            safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT : HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )

        elif "cambrian" in model_id:
            self.conv_mode = "llama_3"
            model_path = f'/scratch/czr/hf_models/{model_id}'
            if overwrite_path is not None:
                model_path = overwrite_path
            self.model_name = get_model_name_from_path(model_path)
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, self.model_name)
            self.temperature = 0


    def generate_response(self, question, video_path):
        if "VideoLLaMA" in self.model_id:
            tensor = process_video(video_path, self.processor, self.model.config.image_aspect_ratio, sample_scheme='uniform').to(dtype=torch.float16, device=self.device, non_blocking=True)
            default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
            modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]
            tensor = [tensor]

            # 3. text preprocess (tag process & generate prompt).
            question = default_mm_token + "\n" + question
            conv = conv_templates_videollama[self.conv_mode].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_MMODAL_token(prompt, self.tokenizer, modal_token_index, return_tensors='pt').unsqueeze(0).to(self.device)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images_or_videos=tensor,
                    modal_list=['video'],
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    use_cache=True,
                )

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            return outputs[0]

        elif "InternVL2" in self.model_id:

            if video_path == None:
                question = question
                response, history = self.model.chat(self.tokenizer, None, question, self.generation_config,
                                            history=None, return_history=True)
            else:
                pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
                pixel_values = pixel_values.to(self.device)
                # pixel_values = pixel_values.to(torch.bfloat16).cuda()
                video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
                question = video_prefix + question
                # Frame1: <image>\nFrame2: <image>\n...\nFrame31: <image>\n{question}
                response, history = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config,
                                            num_patches_list=num_patches_list,
                                            history=None, return_history=True)
            return response
                    
        elif "Chat-UniVi" in self.model_id:

            video_frames, slice_len = _get_rawvideo_dec(video_path, self.image_processor, max_frames=self.max_frames, video_framerate=self.video_framerate)
            qs = question
            cur_prompt = qs
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN * slice_len + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN * slice_len + '\n' + qs

            conv = conv_templates_chatunivi[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=video_frames.half().cuda(),
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    num_beams=self.num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            output_ids = output_ids.sequences
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            return outputs

        elif "MiniCPM" in self.model_id:
            frame, _ = extract_single_frame(video_path)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')

            msgs = [{'role': 'user', 'content': question}]
            
            res = self.model.chat(
                image=image,
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=True,
                temperature=0.7,
            )
            
            generated_text = ""
            for new_text in res:
                generated_text += new_text

            return generated_text

        elif "llava-v1.6-mistral-7b-hf" in self.model_id:
            frame, _ = extract_single_frame(video_path)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')

            prompt_template = "[INST] <image>\n{prompt} [/INST]"
            question = prompt_template.format(prompt=question)

            inputs = self.processor(question, image, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=1280, do_sample=True, temperature=0.7)
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)

            generated_text = generated_text.split("[/INST]")[1].strip()

            return generated_text
        
        elif "gpt-4o" in self.model_id:

            image_size = 768
            max_new_tokens = 200
            sample_frequency = 50

            video = cv2.VideoCapture(video_path)

            # Read the video frame by frame and convert it to base64
            base64Frames = []
            while video.isOpened():
                success, frame = video.read()
                if not success:
                    break
                _, buffer = cv2.imencode(".jpg", frame)
                base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

            video.release()

            sampled_frames = base64Frames[0::sample_frequency]
            max_frame_num = 10
            if len(sampled_frames) > max_frame_num:
                # uniform sampling 20
                sampled_frames = [sampled_frames[i] for i in range(0, len(sampled_frames), len(sampled_frames)//max_frame_num)]
            
            print(len(sampled_frames), "frames read.")

            # Sample the frame from the video, every sample_frequency frames
            video_slice = map(lambda x: {"image": x, "resize": image_size}, sampled_frames)
                    
            # Input the prompt and the video slice to the API
            PROMPT_MESSAGES = [
                {
                    "role": "user",
                    "content": [
                        question,
                        *video_slice,
                    ],
                },
            ]
            
            # Call the API
            params = {
                "model": "gpt-4o",
                "messages": PROMPT_MESSAGES,
                "max_tokens": max_new_tokens,
            }
            result = self.client.chat.completions.create(**params)
            generated_text = result.choices[0].message.content

            return generated_text
    
        elif "gemini" in self.model_id:

            def upload_to_gemini(path, mime_type=None):
                """Uploads the given file to Gemini.

                See https://ai.google.dev/gemini-api/docs/prompting_with_media
                """
                file = self.genai.upload_file(path, mime_type=mime_type)
                print(f"Uploaded file '{file.display_name}' as: {file.uri}")
                return file

            def wait_for_files_active(files):
                """Waits for the given files to be active.

                Some files uploaded to the Gemini API need to be processed before they can be
                used as prompt inputs. The status can be seen by querying the file's "state"
                field.

                This implementation uses a simple blocking polling loop. Production code
                should probably employ a more sophisticated approach.
                """
                print("Waiting for file processing...")
                for name in (file.name for file in files):
                    file = self.genai.get_file(name)
                    while file.state.name == "PROCESSING":
                        print(".", end="", flush=True)
                        time.sleep(10)
                        file = self.genai.get_file(name)
                    if file.state.name != "ACTIVE":
                        raise Exception(f"File {file.name} failed to process")
                print("...all files ready")
                print()

            files = [
                upload_to_gemini(video_path, mime_type="video/mp4"),
            ]
            
            # Some files have a processing delay. Wait for them to be ready.
            wait_for_files_active(files)
            
            contents = [
                files[0],
                question,
            ]
            
            response = self.model.generate_content(contents)
            return response.text


        elif "cambrian" in self.model_id:
            frame, _ = extract_single_frame(video_path)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
            input_ids, image_tensor, image_sizes, prompt = process(image, question, self.tokenizer, self.image_processor, self.model.config, self.conv_mode)
            input_ids = input_ids.to(device=self.device, non_blocking=True)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if self.temperature > 0 else False,
                    temperature=self.temperature,
                    num_beams=1,
                    max_new_tokens=1024,
                    use_cache=True)

            generated_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            return generated_text