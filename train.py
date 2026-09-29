import os
import time
import argparse

import torch

from gliner2 import AutoExtractor
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", default="output")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    model = AutoExtractor.from_pretrained("fastino/GLiNER2.5-multi-Decide", map_location="cpu")

    eval_strategy = "steps" if not args.benchmark else "no"
    config = TrainingConfig(
        output_dir=args.out,
        experiment_name="matugen-jev",
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        task_lr=5e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        scheduler_type="linear",
        warmup_ratio=0.1,
        fp16=False,
        bf16=False,
        max_len=args.max_len,
        eval_strategy=eval_strategy,
        eval_steps=250,
        save_best=True,
        metric_for_best="eval_loss",
        greater_is_better=False,
        early_stopping=True,
        early_stopping_patience=2,
        logging_steps=10,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=2,
        seed=42,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.1,
        lora_target_modules=["encoder", "classifier"],
        save_adapter_only=True,
        on_capacity_exceeded="truncate_with_warning",
        allow_invalid_samples=True,
    )
    trainer = ExtractorTrainer(model, config)
    t0 = time.time()
    results = trainer.train(train_data=args.train, eval_data=args.val)
    print("best_metric:", results.get("best_metric"))
    print("total_steps:", results.get("total_steps"))
    print("total_time_seconds:", results.get("total_time_seconds"))
    print("seconds per step:", results.get("total_time_seconds", 1) / max(results.get("total_steps", 1), 1))


if __name__ == "__main__":
    main()