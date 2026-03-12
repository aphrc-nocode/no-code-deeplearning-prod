# image_classification_train.py

import argparse
import os
# Disable WandB explicitly
os.environ["WANDB_DISABLED"] = "true"
import json
from functools import partial
import torch


from datasets import load_dataset, DatasetDict
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
# Import data utilities
from image_classification_utils.metrics import compute_metrics
from image_classification_utils.data_utils import apply_transforms, collate_fn, random_split
from image_classification_utils.model_utils import load_image_processor, load_model
from image_classification_utils.augmentations import get_train_transform, get_validation_transform
from common_utils import JSONMetricsCallback

def main(args):

    if args.is_presplit:
        dataset = load_dataset("imagefolder", data_dir=args.data_dir)
    else:
        print("Dataset is not pre-split. Loading as a single 'train' split and then splitting.")
        unsplit_dataset = load_dataset("imagefolder", data_dir=args.data_dir, split="train")
        
        train_ds, val_ds, test_ds = random_split(
            unsplit_dataset,
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            random_seed=args.seed
        )
        dataset = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})
    
    labels = dataset["train"].features["label"].names
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    image_processor = load_image_processor(args.model_checkpoint, args.max_image_size)
    
    train_transform = get_train_transform(
        max_image_size=args.max_image_size,
        enable_augmentation=args.enable_augmentation,
        flip_prob=args.flip_prob,
        rotate_limit=args.rotate_limit,
        brightness=args.brightness,
        contrast=args.contrast
    )
    val_transform = get_validation_transform()
    
    # Configure the datasets with dynamic augmentations/transforms
    # This (1) speeds up initialization significantly and (2) ensures random augmentations run per-epoch.
    print("Setting up lazy transforms (on-the-fly)...")
    
    train_dataset = dataset["train"].with_transform(
        partial(apply_transforms, transform=train_transform)
    )
    
    val_dataset = dataset["validation"].with_transform(
        partial(apply_transforms, transform=val_transform)
    )
    
    if "test" in dataset:
        test_dataset = dataset["test"].with_transform(
            partial(apply_transforms, transform=val_transform)
        )
    else:
        test_dataset = None

    
    model = load_model(args.model_checkpoint, id2label, label2id, labels)

    output_dir = os.path.join(args.model_output_root, f"{args.model_checkpoint.split('/')[-1]}-{args.run_name}-{args.version}")



    
    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to="none",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=torch.cuda.is_available(),
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        remove_unused_columns=False, # Must be False so 'image' column reaches the transform!
        seed=args.seed,
        dataloader_num_workers=args.num_proc,
        dataloader_pin_memory=torch.cuda.is_available(),
        optim=args.optimizer,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio if args.warmup_ratio > 0 else (args.warmup_epochs / args.epochs if args.epochs > 0 else 0.0),
    )

    callbacks = [
        JSONMetricsCallback(output_dir=output_dir, metrics_filename=args.metrics_filename),
        EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience, early_stopping_threshold=args.early_stopping_threshold),
    ]
    
    # Build custom feature collator leveraging the loaded processor
    data_collator = partial(collate_fn, image_processor=image_processor)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator, # <-- Use our new collator
        processing_class=image_processor, # Pass processor for saving
        callbacks=callbacks,
    )

    trainer.train()
    
    # Explicitly save model and processor
    trainer.save_model(output_dir)
    image_processor.save_pretrained(output_dir)


        
    if test_dataset:
        test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
        with open(callbacks[0].metrics_file, "a") as f:
            f.write(json.dumps(test_metrics) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an image classification model.")

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_checkpoint", type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--run_name", type=str, default="image-class-run")
    parser.add_argument("--version", type=str, default="1.0.0")
    parser.add_argument("--model_output_root", type=str, default="model_outputs")
    parser.add_argument("--metrics_filename", type=str, default="training_metrics.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_image_size", type=int, default=224)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    # parser.add_argument("--fp16", action="store_true") # Auto-detected
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)

    
    # Augmentation (Added Phase 8)
    parser.add_argument("--enable_augmentation", action="store_true")
    parser.add_argument("--flip_prob", type=float, default=0.5)
    parser.add_argument("--rotate_limit", type=int, default=0)
    parser.add_argument("--brightness", type=float, default=0.2)
    parser.add_argument("--contrast", type=float, default=0.2)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--is_presplit", action="store_true", help="Whether the dataset is already split into train/validation/test folders.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio if not pre-split.")
    parser.add_argument("--dev_ratio", type=float, default=0.1, help="Validation split ratio if not pre-split.")
    
    # Advanced Training Params (Added for Production Audit)
    parser.add_argument("--optimizer", type=str, default="adamw_torch", help="Optimizer (adamw_torch, sgd, adafactor)")
    parser.add_argument("--scheduler", type=str, default="linear", help="LR Scheduler (linear, cosine, constant)")
    parser.add_argument("--warmup_epochs", type=float, default=0.0, help="Warmup epochs (converted to ratio)")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Warmup ratio (overrides epochs)")
    
    args = parser.parse_args()
    main(args)