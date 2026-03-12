# image_segmentation_train.py

import argparse
import os
# Disable WandB explicitly
os.environ["WANDB_DISABLED"] = "true"
import json
from functools import partial

import torch
import glob

from datasets import load_dataset, DatasetDict, Image
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)


# Import our new segmentation utils
from image_segmentation_utils.metrics import create_compute_metrics_fn
# Import data processing utilities
from image_segmentation_utils.data_utils import apply_transforms, collate_fn, random_split
from image_segmentation_utils.model_utils import load_image_processor, load_model
from image_segmentation_utils.augmentations import get_train_transform, get_validation_transform
from common_utils import JSONMetricsCallback

import torch.nn as nn

class SmpTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation for SMP models.
        """
        # 1. Unpack Inputs
        # The collator returns {"pixel_values": ..., "labels": ...}
        images = inputs["pixel_values"]
        labels = inputs["labels"]
        
        # 2. Forward Pass
        # SMP models just take 'x'
        logits = model(images)
        
        # 3. Compute Loss
        # Hugging Face models usually compute loss internally. SMP does not.
        # labels are (B, H, W) -> Long
        # logits are (B, NumClasses, H, W)
        
        # Resize logits to match label size if needed (though usually we resize inputs)
        # Assuming inputs are already resized by aug pipeline.
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=255) # 255 is common ignore index
        
        # Ensure labels are 3D (B, H, W)
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
            
        loss = loss_fct(logits, labels)
        
        return (loss, logits) if return_outputs else loss

def get_trainer_class(model):
    """
    Returns the appropriate Trainer class.
    """
    # Simple heuristic: if it has 'encoder' attribute, it's likely SMP
    if hasattr(model, "encoder") and hasattr(model, "decoder"):
        return SmpTrainer
    return Trainer

def main(args):
    
    # --- 1. Robust Data Loading & Label Detection ---
    
    # 1.1 Helper to scan directory for unique mask values
    def get_unique_labels_from_masks(mask_paths, sample_size=50):
        print(f"Auto-detecting labels from {len(mask_paths)} masks (sampling {sample_size})...")
        unique_labels = set()
        import random
        from PIL import Image as PILImage
        import numpy as np
        
        # Sample random masks
        sample = random.sample(mask_paths, min(len(mask_paths), sample_size))
        for p in sample:
            try:
                # Open as PIL, convert to numpy array
                img = PILImage.open(p)
                arr = np.array(img)
                unique_labels.update(np.unique(arr))
            except Exception as e:
                print(f"Warning: Failed to read mask {p}: {e}")
                
        # Filter out 255 (common ignore index) if present, unless it's a real class
        # Usually 255 is ignore. Let's keep it for now and let user decide or assume it's background if 0 missing.
        print(f"Found unique pixel values: {sorted(list(unique_labels))}")
        return sorted(list(unique_labels))

    # 1.2 Helper to pair images and masks
    def pair_images_and_masks(split_dir):
        image_dir = os.path.join(split_dir, "images")
        mask_dir = os.path.join(split_dir, "masks")
        
        if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
            return [], []
            
        # Get all files (ignoring extensions for matching)
        img_files = sorted(glob.glob(os.path.join(image_dir, "*")))
        mask_files = sorted(glob.glob(os.path.join(mask_dir, "*")))
        
        # Map stems to full paths
        img_map = {os.path.splitext(os.path.basename(f))[0]: f for f in img_files}
        mask_map = {os.path.splitext(os.path.basename(f))[0]: f for f in mask_files}
        
        final_images = []
        final_masks = []
        
        # Intersect keys
        common_stems = sorted(list(set(img_map.keys()) & set(mask_map.keys())))
        
        for stem in common_stems:
            final_images.append(img_map[stem])
            final_masks.append(mask_map[stem])
            
        print(f"Found {len(final_images)} paired images/masks in {split_dir}")
        return final_images, final_masks

    # --- Step A: Check for metadata.json ---
    metadata_path = os.path.join(args.data_dir, "metadata.json")
    json_files = glob.glob(os.path.join(args.data_dir, "*.json"))
    valid_jsons = [f for f in json_files if not os.path.basename(f).startswith(".")]
    
    id2label = {}
    label2id = {}
    
    if valid_jsons:
        metadata_path = valid_jsons[0]
        print(f"Using metadata file: {metadata_path}")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        id2label = {int(k): v for k, v in metadata["id2label"].items()}
        label2id = {v: int(k) for k, v in id2label.items()}
    else:
        print("No metadata.json found. Will auto-detect labels from masks later.")

    # --- Step B: Load Dataset (Manual Scan or ImageFolder) ---
    dataset = DatasetDict()
    
    if args.is_presplit:
        # Check standard structure
        splits = ["train", "validation", "test"]
        found_splits = []
        
        all_mask_paths = []
        
        for split in splits:
            split_dir = os.path.join(args.data_dir, split)
            if os.path.isdir(split_dir):
                imgs, masks = pair_images_and_masks(split_dir)
                if imgs:
                    # Create HF Dataset
                    d = Dataset.from_dict({"image": imgs, "label": masks})
                    d = d.cast_column("image", Image())
                    d = d.cast_column("label", Image(decode=False)) # Keep raw entries for mask
                    dataset[split] = d
                    found_splits.append(split)
                    if split == "train":
                        all_mask_paths.extend(masks)
        
        if not found_splits:
             # Fallback to imagefolder if structure is weird but metadata exists?
             pass
    else:
        # Not presplit: assume root images/ and masks/
        imgs, masks = pair_images_and_masks(args.data_dir)
        if imgs:
            full_ds = Dataset.from_dict({"image": imgs, "label": masks})
            full_ds = full_ds.cast_column("image", Image())
            full_ds = full_ds.cast_column("label", Image(decode=False))
            all_mask_paths.extend(masks)
            
            # Split
            train_ds, val_ds, test_ds = random_split(
                full_ds,
                train_ratio=args.train_ratio,
                dev_ratio=args.dev_ratio,
                random_seed=args.seed
            )
            dataset = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})

    # --- Step C: Finalize Labels ---
    if not id2label:
        if not dataset:
             raise ValueError("Failed to load any valid image/mask pairs from dataset directory.")
             
        # Auto-detect from 'train' masks (or all masks found)
        # We need mask paths. If we loaded via Dataset.from_dict, we have them in the 'label' column (as paths? no, Cast(Image) makes them objects?)
        # Wait, Image(decode=False) keeps them as dicts with 'path' and 'bytes'.
        
        # Actually, let's use the 'all_mask_paths' list we collected during scanning.
        sample_masks = []
        if "train" in dataset:
             # If we used cast_column, accessing dataset['train'][0]['label'] returns a dict with 'path'.
             # But 'all_mask_paths' variable above populated only if we used Manual pairing.
             pass
        
        # For simplicity, if we used Manual Pairing, 'all_mask_paths' is populated.
        # If we fell back (unlikely here), we might need to iterate dataset.
        
        unique_vals = get_unique_labels_from_masks(all_mask_paths)
        if not unique_vals or len(unique_vals) <= 1:
            raise ValueError(
                f"Failed to auto-detect valid classes from masks. Found: {unique_vals}. "
                "Semantic segmentation requires at least 2 classes (e.g., Background + Object). "
                "Please verify your dataset mask images or provide a metadata.json file."
            )
            
        print(f"Auto-detected {len(unique_vals)} classes: {unique_vals}")
        
        # Construct id2label
        for val in unique_vals:
             id2label[int(val)] = f"class_{val}"
        label2id = {v: k for k, v in id2label.items()}
            
    num_labels = len(id2label)
    print(f"Final Configuration: {num_labels} labels: {id2label}")

    # --- 3. Load Model & Processor ---
    image_processor = load_image_processor(args.model_checkpoint, args.max_image_size)
    model = load_model(args.model_checkpoint, id2label, label2id)

    # --- 4. Define Transforms ---
    train_transform = get_train_transform(
        max_image_size=args.max_image_size,
        enable_augmentation=args.enable_augmentation,
        flip_prob=args.flip_prob,
        rotate_limit=args.rotate_limit,
        brightness=args.brightness,
        contrast=args.contrast
    )
    val_transform = get_validation_transform(args.max_image_size)
    
    # Map preprocessing function to speed up training pipeline
    print("Applying transforms to train dataset...")
    train_dataset = dataset["train"].map(
        partial(apply_transforms, transform=train_transform),
        batched=True,
        num_proc=args.num_proc,
        remove_columns=["image", "label"] # Remove original columns
    )
    
    print("Applying transforms to validation dataset...")
    val_dataset = dataset["validation"].map(
        partial(apply_transforms, transform=val_transform),
        batched=True,
        num_proc=args.num_proc,
        remove_columns=["image", "label"]
    )
    
    if "test" in dataset:
        print("Applying transforms to test dataset...")
        test_dataset = dataset["test"].map(
            partial(apply_transforms, transform=val_transform),
            batched=True,
            num_proc=args.num_proc,
            remove_columns=["image", "label"]
        )
    else:
        test_dataset = None


    # --- 5. Setup Trainer ---
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.model_output_root, f"{args.model_checkpoint.split('/')[-1]}-{args.run_name}-{args.version}")


    
    compute_metrics_fn = create_compute_metrics_fn(num_labels, id2label)
    
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
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_mean_iou",
        greater_is_better=True,
        remove_unused_columns=False, # We already manually removed columns
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
    
    # Use custom default metric collator
    data_collator = collate_fn
    
    # Select default or SMP trainer
    TrainerClass = get_trainer_class(model)
    print(f"Using Trainer Class: {TrainerClass.__name__}")
    
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn,
        data_collator=data_collator, # <-- Use our new collator
        processing_class=image_processor, 
        callbacks=callbacks,
    )

    trainer.train()
    
    # Explicitly save model and processor
    trainer.save_model(output_dir)
    image_processor.save_pretrained(output_dir)


        
    if test_dataset is not None:
        print("\n--- Evaluating on Test Set ---")
        test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
        print(test_metrics)
        with open(callbacks[0].metrics_file, "a") as f:
            f.write(json.dumps(test_metrics) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an image segmentation model.")

    # Paths and Naming
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_checkpoint", type=str, default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--run_name", type=str, default="seg-run")
    parser.add_argument("--version", type=str, default="1.0.0")
    parser.add_argument("--model_output_root", type=str, default="model_outputs")
    parser.add_argument("--output_dir", type=str, default=None, help="Explicit output directory. Overrides model_output_root+run_name logic.")
    parser.add_argument("--metrics_filename", type=str, default="training_metrics.json")
    
    # Data Splitting
    parser.add_argument("--is_presplit", action="store_true", help="Whether the dataset is already split into train/validation/test folders.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio if not pre-split.")
    parser.add_argument("--dev_ratio", type=float, default=0.1, help="Validation split ratio if not pre-split.")

    # Training Parameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_image_size", type=int, default=512)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Execution & Reproducibility
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    # parser.add_argument("--fp16", action="store_true") # Auto-detected

    # Saving & Early Stopping
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)
    
    # Hub & Logging

    
    # Augmentation (Added Phase 6)
    parser.add_argument("--enable_augmentation", action="store_true")
    parser.add_argument("--flip_prob", type=float, default=0.5)
    parser.add_argument("--rotate_limit", type=int, default=0)
    parser.add_argument("--brightness", type=float, default=0.2)
    parser.add_argument("--contrast", type=float, default=0.2)
    
    # Advanced Training Params (Added for Production Audit)
    parser.add_argument("--optimizer", type=str, default="adamw_torch", help="Optimizer (adamw_torch, sgd, adafactor)")
    parser.add_argument("--scheduler", type=str, default="linear", help="LR Scheduler (linear, cosine, constant)")
    parser.add_argument("--warmup_epochs", type=float, default=0.0, help="Warmup epochs (converted to ratio)")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Warmup ratio (overrides epochs)")


    args = parser.parse_args()
    main(args)