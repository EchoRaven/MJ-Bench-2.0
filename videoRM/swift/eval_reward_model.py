# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Tuple

from swift.llm.utils import get_model_with_value_head
from swift.trainers import TrainerFactory
from swift.tuners import Swift
from swift.utils import get_logger, get_main, seed_everything
from swift.llm.sft import prepare_dataset, prepare_model_template_train
from swift.llm.utils import TEMPLATE_MAPPING, RLHFArguments
from swift.trainers.rlhf_trainer.reward_trainer import RewardTrainer
import os
from functools import partial
from accelerate.utils import gather_object
import json
import torch
import transformers
from datasets import Dataset as HfDataset
from packaging import version
from transformers import BitsAndBytesConfig, GenerationConfig, IntervalStrategy
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils import is_torch_npu_available, strtobool

from swift.torchacc_utils import patch_acc_model
from swift.trainers.utils import can_return_loss, find_labels
from swift.utils import (append_to_jsonl, check_json_format, compute_acc_metrics, compute_nlg_metrics, get_dist_setting,
                         get_logger, get_main, get_model_info, is_ddp_plus_mp, is_dist, is_master, plot_images,
                         preprocess_logits_for_metrics, seed_everything, show_layers, use_torchacc)
from swift.llm.accelerator import ta_accelerate
from swift.llm.tuner import prepare_model
from swift.llm.utils import (TEMPLATE_MAPPING, LazyLLMDataset, PtArguments, RLHFArguments, SftArguments, Template, dataset_map,
                    deep_getattr, dynamic_vit_gradient_checkpointing, get_dataset, get_mllm_arch, get_model_tokenizer,
                    get_template, get_time_info, print_example, set_generation_config, sort_by_max_length, stat_dataset, get_model_with_value_head)
from transformers import AutoModelForCausalLM
from peft import PeftModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, PreTrainedModelWrapper
from safetensors.torch import load_file, save_file
from safetensors import safe_open

def load_safetensor(model_path):
    tensors = {}
    with safe_open(model_path, framework="pt", device='cpu') as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors

logger = get_logger()

json_file_name = "cc_reward_score.json"
json_data = []

def trainer_train(
    args,
    model,
    template,
    train_dataset,
    val_dataset,
    callbacks=None,
    msg=None,
    ref_model=None,
    reward_model=None,
    value_model=None,
) -> Dict[str, Any]:
    if msg is None:
        msg = {}
    training_args = args.training_args
    padding_to = args.max_length if args.sft_type == 'longlora' else None
    tokenizer = template.tokenizer
    data_collator = partial(template.data_collator, padding_to=padding_to)

    if use_torchacc():
        train_batch_size = args.batch_size
        eval_batch_size = args.eval_batch_size
        train_batch_size *= args.world_size
        eval_batch_size *= args.world_size
        training_args.per_device_train_batch_size = train_batch_size
        training_args.per_device_eval_batch_size = eval_batch_size
        training_args.group_by_length = use_torchacc()

    logger.info(f'training_args: {training_args}')

    trainer_cls, trainer_kwargs = TrainerFactory.get_trainer_info(args)
    if not hasattr(model.config, 'is_encoder_decoder'):
        model.config.is_encoder_decoder = False
    is_encoder_decoder = model.config.is_encoder_decoder
    trainer_kwargs['is_encoder_decoder'] = is_encoder_decoder
    if args.check_model_is_latest is False:
        trainer_kwargs['check_model'] = False
    if isinstance(args, RLHFArguments):
        trainer_kwargs['ref_model'] = ref_model
    elif args.predict_with_generate:
        trainer_kwargs['compute_metrics'] = partial(compute_nlg_metrics, tokenizer=tokenizer)
    else:
        compute_metrics = partial(
            compute_acc_metrics, acc_strategy=args.acc_strategy, is_encoder_decoder=is_encoder_decoder)
        trainer_kwargs['compute_metrics'] = compute_metrics
        trainer_kwargs['preprocess_logits_for_metrics'] = preprocess_logits_for_metrics
    if args.train_type == 'ppo':
        trainer_kwargs['reward_model'] = reward_model
        trainer_kwargs['value_model'] = value_model
    trainer = trainer_cls(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks,
        **trainer_kwargs)
    trainer.is_multimodal = args.is_multimodal
    trainer.sft_args = args
    if use_torchacc():
        trainer.label_names = model.label_names
        trainer.can_return_loss = model.return_loss
    if is_master():
        for args_obj, fname in zip([args, training_args], ['sft_args.json', 'training_args.json']):
            fpath = os.path.join(args.output_dir, fname)
            logger.info(f'The {args_obj.__class__.__name__} will be saved in: {fpath}')
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(check_json_format(args_obj.__dict__), f, ensure_ascii=False, indent=2)
    logging_path = os.path.join(args.output_dir, 'logging.jsonl')
    logger.info(f'The logging file will be saved in: {logging_path}')
    with template.training_context():
        trainer.train(training_args.resume_from_checkpoint)
    last_model_checkpoint = getattr(trainer.state, 'last_model_checkpoint', None)
    logger.info(f'last_model_checkpoint: {last_model_checkpoint}')
    logger.info(f'best_model_checkpoint: {trainer.state.best_model_checkpoint}')
    # Visualization
    if is_master() and not use_torchacc():
        if 'tensorboard' in training_args.report_to:
            images_dir = os.path.join(args.output_dir, 'images')
            logger.info(f'images_dir: {images_dir}')
            plot_images(images_dir, args.logging_dir, ['train/loss'], 0.9)
        if args.push_to_hub:
            trainer.push_to_hub()
    run_info = {
        'memory': trainer.perf['memory'],
        'last_model_checkpoint': last_model_checkpoint,
        'best_model_checkpoint': trainer.state.best_model_checkpoint,
        'best_metric': trainer.state.best_metric,
        'global_step': trainer.state.global_step,
        'log_history': trainer.state.log_history,
        **msg
    }
    if not args.streaming:
        train_time = get_time_info(trainer.state.log_history, len(train_dataset))
        run_info.update({'train_time': train_time})
    for key in ['gen_time', 'gen_len']:
        if key in trainer.perf and trainer.perf[key] != 0:
            run_info[key] = trainer.perf[key]
    if is_master():
        jsonl_path = os.path.join(args.output_dir, 'logging.jsonl')
        append_to_jsonl(jsonl_path, run_info)
    return trainer

def llm_rlhf(args: RLHFArguments) -> Dict[str, Any]:
    logger.info(f'args: {args}')
    seed_everything(args.seed)

    is_generation = TEMPLATE_MAPPING[args.template_type].get('is_generation', False)
    if is_generation:
        logger.warning(f"Please check if args.template_type: '{args.template_type}' is correct.")

    kwargs = {}
    if args.rlhf_type == 'ppo':
        from copy import deepcopy
        reward_model_args, value_model_args = deepcopy(args), deepcopy(args)
        args_to_modified = ['model_id_or_path', 'model_type', 'model_revision']
        for model_args in [reward_model_args, value_model_args]:
            for arg in args_to_modified:
                setattr(model_args, arg, getattr(args, f'reward_{arg}'))
        reward_model_args.ref_model_free = True  # avoid to create ref model
        value_model_args.ref_model_free = True
        reward_model, _, _, _ = prepare_model_template_train(reward_model_args)
        reward_model.requires_grad_(False).eval()

        reward_model = get_model_with_value_head(reward_model)  # add and load value head
        # hack here to customize the value model
        value_model, _, _, _ = prepare_model_template_train(value_model_args)
        value_model = get_model_with_value_head(value_model)
        kwargs['reward_model'] = reward_model
        kwargs['value_model'] = value_model

    msg = {}
    model, ref_model, template, callbacks = prepare_model_template_train(args)

    with TrainerFactory.patch_template(args, template):
        train_dataset, val_dataset = prepare_dataset(args, template, msg)

        trainer = trainer_train(
            args,
            model,
            template,
            train_dataset,
            val_dataset,
            callbacks=callbacks,
            msg=msg,
            ref_model=ref_model,
            **kwargs)
        with open(json_file_name, "w", encoding="utf-8") as f:
            with template.training_context():
                eval_dataloader = trainer.get_eval_dataloader()
                for _, inputs in enumerate(eval_dataloader):
                    _, logits, _ = trainer.prediction_step(trainer.model, inputs, prediction_loss_only=False)
                    batch_size = inputs['input_ids'].shape[0] // 2
                    text = trainer.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
                    chosen_text, rejected_text = text[:batch_size], text[batch_size:]
                    json_data.append(
                        {
                            "chosen": {
                                "text": chosen_text[0],
                                "score": logits[0][0].item()
                            },
                            "reject": {
                                "text": rejected_text[0],
                                "score": logits[0][1].item()
                            }
                        }
                    )
            json.dump(json_data, f, indent=4, ensure_ascii=False)
                

rlhf_main = get_main(RLHFArguments, llm_rlhf)

if __name__ == "__main__":
    rlhf_main()
