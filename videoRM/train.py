# train.py
import argparse
import torch
from transformers import Trainer, TrainingArguments
from dataset.load_dataset import load_saved_data
from transformers import AutoModel
from torch.utils.tensorboard import SummaryWriter  # 用于TensorBoard

class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tb_writer = SummaryWriter(self.args.logging_dir)  # 初始化 TensorBoard SummaryWriter

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids_0 = inputs['input_ids_0']
        attention_mask_0 = inputs['attention_mask_0']
        pixel_values_0 = inputs['pixel_values_0']
        labels_0 = inputs['labels_0']

        input_ids_1 = inputs['input_ids_1']
        attention_mask_1 = inputs['attention_mask_1']
        pixel_values_1 = inputs['pixel_values_1']
        labels_1 = inputs['labels_1']

        # 如果 image_flags_0 不存在，生成全1的 image_flags
        if 'image_flags_0' not in inputs:
            image_flags_0 = torch.ones(pixel_values_0.shape[0], dtype=torch.long)
        else:
            image_flags_0 = inputs['image_flags_0']

        # 如果 image_flags_1 不存在，生成全1的 image_flags
        if 'image_flags_1' not in inputs:
            image_flags_1 = torch.ones(pixel_values_1.shape[0], dtype=torch.long)
        else:
            image_flags_1 = inputs['image_flags_1']

        # 对 group 0 进行前向传播
        outputs_0 = model(
            input_ids=input_ids_0,
            attention_mask=attention_mask_0,
            pixel_values=pixel_values_0,
            labels=labels_0,
            image_flags=image_flags_0  # 传递 image_flags
        )

        # 对 group 1 进行前向传播
        outputs_1 = model(
            input_ids=input_ids_1,
            attention_mask=attention_mask_1,
            pixel_values=pixel_values_1,
            labels=labels_1,
            image_flags=image_flags_1  # 传递 image_flags
        )

        loss_0 = outputs_0.loss
        loss_1 = outputs_1.loss
        loss = (loss_0 + loss_1) / 2

        # 记录损失到 TensorBoard
        self.tb_writer.add_scalar("Loss/group_0", loss_0.item(), self.state.global_step)
        self.tb_writer.add_scalar("Loss/group_1", loss_1.item(), self.state.global_step)
        self.tb_writer.add_scalar("Loss/combined", loss.item(), self.state.global_step)

        if return_outputs:
            return loss, (outputs_0, outputs_1)
        return loss

    def log(self, logs: dict):
        super().log(logs)
        if self.tb_writer:
            for key, value in logs.items():
                self.tb_writer.add_scalar(key, value, self.state.global_step)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with custom arguments")

    parser.add_argument('--path', type=str, default="OpenGVLab/InternVL2-2B", help="Pretrained model path")
    parser.add_argument('--train_save_dir', type=str, default="dataset/saved_train_dataset", help="Path to saved train dataset")
    parser.add_argument('--test_save_dir', type=str, default="dataset/saved_test_dataset", help="Path to saved test dataset")
    parser.add_argument('--output_dir', type=str, default="./results", help="Directory to save the results")
    parser.add_argument('--logging_dir', type=str, default="./logs", help="Directory to save logs")
    parser.add_argument('--epochs', type=int, default=3, help="Number of epochs to train")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training and evaluation")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    parser.add_argument('--warmup_steps', type=int, default=500, help="Number of warmup steps")
    parser.add_argument('--logging_steps', type=int, default=10, help="Logging steps during training")
    parser.add_argument('--evaluation_strategy', type=str, default="epoch", help="Evaluation strategy (epoch, steps, etc.)")

    return parser.parse_args()

def main():
    args = parse_args()

    model = AutoModel.from_pretrained(
        args.path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
    ).cuda()

    train_dataset, test_dataset = load_saved_data(args.train_save_dir, args.test_save_dir)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy=args.evaluation_strategy,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_dir=args.logging_dir,  # 指定TensorBoard的日志目录
        logging_steps=args.logging_steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        report_to="tensorboard",  # 启用TensorBoard
        remove_unused_columns=False
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    main()
