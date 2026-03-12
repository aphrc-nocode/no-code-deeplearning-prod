# object_detection_train.py

import os
import os
# Disable WandB integration
os.environ["WANDB_DISABLED"] = "true"

import argparse
import json
import sys
import subprocess
from functools import partial

import datasets
import torch

from transformers import TrainingArguments, Trainer, TrainerCallback, EarlyStoppingCallback
from object_detection_utils.data_utils import augment_and_transform_batch, filter_invalid_objects_coco
from object_detection_utils.augmentations import get_train_transform, get_validation_transform
from object_detection_utils.model_utils import load_model, load_image_processor, collate_fn
from object_detection_utils.metrics import compute_metrics
# Import from common_utils
from common_utils import JSONMetricsCallback

# --- REMOVED: Local JSONMetricsCallback class definition ---

def main(args):
    """Main function to run the training and evaluation."""
    
    # Auto-detect if input is already an Arrow dataset
    # Convert to absolute path to be safe
    abs_data_dir = os.path.abspath(args.data_dir)
    # Check for Hugging Face Arrow format indicators
    has_dataset_dict = os.path.exists(os.path.join(abs_data_dir, "dataset_dict.json"))
    has_train_arrow = os.path.exists(os.path.join(abs_data_dir, "train", "state.json")) or \
                      os.path.exists(os.path.join(abs_data_dir, "train", "dataset_info.json"))
    
    if has_dataset_dict or has_train_arrow:
        print(f"Detected existing Arrow dataset at {args.data_dir}. Skipping preprocessing.")
        args.processed_data_dir = args.data_dir
    
    if not os.path.exists(args.processed_data_dir) or args.force_preprocess:
        print("Processed dataset not found or reprocessing forced.")
        print("Running preprocessing script...")
        
        preprocess_script_path = os.path.join(
            os.path.dirname(__file__), "object_detection_utils", "preprocess_data.py"
        )
        
        subprocess.run([
            sys.executable, preprocess_script_path,
            "--raw_data_dir", args.data_dir,
            "--processed_data_dir", args.processed_data_dir
        ], check=True)
    else:
        print(f"Found existing processed dataset at {args.processed_data_dir}")

    print("Loading processed dataset from disk...")
    dataset = datasets.load_from_disk(args.processed_data_dir)
    
    output_dir = os.path.join(args.model_output_root, f"{args.model_checkpoint.split('/')[-1]}-{args.run_name}-{args.version}")
    
    dataset = dataset.map(filter_invalid_objects_coco, num_proc=args.num_proc)

    categories = dataset["train"].features["objects"]["category"].feature.names
    
    id2label = {i: name for i, name in enumerate(categories)}
    label2id = {v: k for k, v in id2label.items()}
    
    image_processor = load_image_processor(args.model_checkpoint, args.max_image_size)
    model = load_model(args.model_checkpoint, id2label, label2id)

    aug_config = {
        "enable_augmentation": args.enable_augmentation,
        "flip_prob": args.flip_prob,
        "rotate_limit": args.rotate_limit,
        "brightness": args.brightness,
        "contrast": args.contrast
    }
    
    train_transform = get_train_transform(args.max_image_size, config=aug_config)
    val_transform = get_validation_transform()
    
    train_transform_batch = partial(augment_and_transform_batch, transform=train_transform, image_processor=image_processor)
    validation_transform_batch = partial(augment_and_transform_batch, transform=val_transform, image_processor=image_processor)

    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    dataset["validation"] = dataset["validation"].with_transform(validation_transform_batch)
    test_dataset = dataset["test"].with_transform(validation_transform_batch)

    eval_compute_metrics_fn = partial(compute_metrics, image_processor=image_processor, id2label=id2label)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to="none",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        logging_strategy="epoch",
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optim=args.optimizer,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio if args.warmup_ratio > 0 else (args.warmup_epochs / args.epochs if args.epochs > 0 else 0.0),
        save_total_limit=1,
        metric_for_best_model="eval_map",
        greater_is_better=True,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        dataloader_num_workers=args.num_proc,
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=args.seed
    )
    
    json_metrics_callback = JSONMetricsCallback(output_dir=output_dir, metrics_filename=args.metrics_filename)
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=image_processor,
        callbacks=[json_metrics_callback, early_stopping_callback]
    )

    trainer.train()

    # Explicitly save model
    trainer.save_model(output_dir)
    image_processor.save_pretrained(output_dir)



    test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
    
    with open(json_metrics_callback.metrics_file, "a") as f:
        f.write(json.dumps(test_metrics) + "\n")

    print("\n--- Training and Evaluation Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an object detection model.")
    
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the RAW data directory (from unzipped file).")
    parser.add_argument("--processed_data_dir", type=str, required=True, help="Path to store/load the processed Arrow dataset.")
    parser.add_argument("--force_preprocess", action="store_true", help="Force reprocessing of the dataset even if it exists.")
    parser.add_argument("--metrics_filename", type=str, default="training_metrics.json")
    parser.add_argument("--model_output_root", type=str, default="model_outputs")
    parser.add_argument("--model_checkpoint", type=str, default="facebook/detr-resnet-50")
    parser.add_argument("--run_name", type=str, default="detr-finetune")
    parser.add_argument("--version", type=str, default="0.0")
    parser.add_argument("--max_image_size", type=int, default=600)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    # parser.add_argument("--max_grad_norm",type=float, default=1.0) # Removed, use Trainer default
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=4)
    # parser.add_argument("--fp16", action="store_true") # Removed, auto-detect
    parser.add_argument("--early_stopping_patience", type=int, default=5)

    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)
    
    # Advanced Training Params (Added for Production Audit)
    parser.add_argument("--optimizer", type=str, default="adamw_torch", help="Optimizer (adamw_torch, sgd, adafactor)")
    parser.add_argument("--scheduler", type=str, default="linear", help="LR Scheduler (linear, cosine, constant)")
    parser.add_argument("--warmup_epochs", type=float, default=0.0, help="Warmup epochs (converted to ratio)")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Warmup ratio (overrides epochs)")
    
    # Augmentation
    parser.add_argument("--enable_augmentation", action="store_true")
    parser.add_argument("--flip_prob", type=float, default=0.5)
    parser.add_argument("--rotate_limit", type=float, default=0.0)
    parser.add_argument("--brightness", type=float, default=0.2)
    parser.add_argument("--contrast", type=float, default=0.2)
    
    args = parser.parse_args()
    main(args)