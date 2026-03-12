# object_detection_train_yolo.py

import argparse
import os
# Disable WandB explicitly
os.environ["WANDB_DISABLED"] = "true"
import json
import yaml
import shutil
import datasets
import torch
from ultralytics import YOLO
from ultralytics.engine.model import Model
from ultralytics.engine.trainer import BaseTrainer
from typing import Dict, Any

# We must import our central callback utility
from common_utils import JSONMetricsCallback
from object_detection_utils.yolo_utils import prepare_yolo_data



# --- REVISED Custom Ultralytics Callback ---

class UltralyticsJSONCallback:
    """
    Custom callback to hook into Ultralytics's events and log to our JSONL file.
    
    This revised version correctly parses the metrics dictionary provided
    by the Ultralytics trainer, logging training losses and validation mAP.
    """
    def __init__(self, output_dir, metrics_filename="training_metrics.json"):
        self.output_dir = output_dir
        self.metrics_file = os.path.join(output_dir, metrics_filename)
        os.makedirs(self.output_dir, exist_ok=True)
        if os.path.exists(self.metrics_file):
            os.remove(self.metrics_file) # Start fresh for each run
        print(f"JSONMetricsCallback initialized. Logging to: {self.metrics_file}")

    def __call__(self, trainer: BaseTrainer):
        """
        This is called on the 'on_fit_epoch_end' event (after validation).
        
        --- V6 FIX ---
        This version correctly handles trainer.tloss as a 3-element tensor.
        We sum it to get the total loss and access its indices for individual losses.
        """
        log_data = {}
        log_data["epoch"] = int(trainer.epoch)

        # --- 1. Get Training Losses (from trainer.tloss) ---
        if trainer.tloss is not None:
            try:
                # trainer.tloss is a 3-element tensor [box_loss, cls_loss, dfl_loss]
                

                total_loss = trainer.tloss.sum().item()
                log_data["train_loss"] = round(float(total_loss), 5)
                log_data["train_box_loss"] = round(float(trainer.tloss[0].item()), 5) if len(trainer.tloss) > 0 else 0.0
                log_data["train_cls_loss"] = round(float(trainer.tloss[1].item()), 5) if len(trainer.tloss) > 1 else 0.0
                log_data["train_dfl_loss"] = round(float(trainer.tloss[2].item()), 5) if len(trainer.tloss) > 2 else 0.0

            except Exception as e:
                print(f"Callback Error: Failed to parse trainer.tloss. Error: {e}")
                log_data["eval_loss"] = 0.0
                log_data["eval_box_loss"] = 0.0
                log_data["eval_cls_loss"] = 0.0
                log_data["eval_dfl_loss"] = 0.0
        else:
            print("Callback Warning: trainer.tloss is None. Logging losses as 0.0")
            log_data["eval_loss"] = 0.0
            log_data["eval_box_loss"] = 0.0
            log_data["eval_cls_loss"] = 0.0
            log_data["eval_dfl_loss"] = 0.0

        # --- 2. Get Validation Metrics (This part is correct) ---
        if trainer.validator and hasattr(trainer.validator, "metrics") and trainer.validator.metrics:
            results_dict = trainer.validator.metrics.results_dict
        else:
            print("Callback Warning: No validator metrics found.")
            results_dict = {}

        # Map Ultralytics validation keys
        key_map = {
            "metrics/precision(B)": "eval_precision_B",
            "metrics/recall(B)": "eval_recall_B",
            "metrics/mAP50(B)": "eval_map_50",
            "metrics/mAP50-95(B)": "eval_map",
            "fitness": "eval_fitness",
            "val/box_loss": "eval_box_loss",
            "val/cls_loss": "eval_cls_loss",
            "val/dfl_loss": "eval_dfl_loss"
        }

        # Process all keys from the map (from the validation results_dict)
        for key, value in results_dict.items():
            if key in key_map:
                log_data[key_map[key]] = round(float(value), 5)
        
        print(f"Logging metrics for epoch {log_data['epoch']}: {log_data}")

        try:
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(log_data) + "\n")
        except Exception as e:
            print(f"Error writing to metrics file: {e}")

            
    def log_test_metrics(self, metrics: Dict[str, Any]):
        """
        Logs the final test metrics.
        """
        log_data = {}
        results_dict = metrics.results_dict
        
        key_map = {
            "metrics/precision(B)": "test_precision_B",
            "metrics/recall(B)": "test_recall_B",
            "metrics/mAP50(B)": "test_map_50",
            "metrics/mAP50-95(B)": "test_map"
        }

        for key, value in results_dict.items():
            if key in key_map:
                log_data[key_map[key]] = round(float(value), 5)
        
        print(f"Logging final test metrics: {log_data}")
        try:
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(log_data) + "\n")
        except Exception as e:
            print(f"Error writing to metrics file: {e}")

# --- Main Training Function ---

def main(args):
    """Main function to run the YOLO training and evaluation."""
    
    print(f"Loading processed dataset from disk: {args.data_dir}")
    dataset = datasets.load_from_disk(args.data_dir)
    
    # --- PHASE 16: Use Optimized Cache ---
    data_yaml_path, class_names = prepare_yolo_data(
        dataset=dataset, 
        data_dir=args.data_dir, # Cache is stored inside data_dir/yolo_cache
        output_dir=args.output_dir
    )
    
    # Check if test split exists
    has_test_split = "test" in dataset
    
    print(f"Loading YOLO model: {args.model_checkpoint}")
    model = YOLO(args.model_checkpoint)
    
    json_metrics_callback = UltralyticsJSONCallback(
        output_dir=args.output_dir, 
        metrics_filename=args.metrics_filename
    )
    model.add_callback("on_fit_epoch_end", json_metrics_callback)
    
    print("Starting YOLO training...")
    device = 0 if torch.cuda.is_available() else "cpu"
    
    # Extract all hyperparameters
    model.train(
        data=data_yaml_path,
        epochs=args.epochs,
        batch=args.train_batch_size,
        imgsz=args.max_image_size,
        workers=args.num_proc,
        project=os.path.abspath(os.path.dirname(args.output_dir)), 
        name=os.path.basename(args.output_dir), 
        exist_ok=True,
        seed=args.seed,
        patience=args.early_stopping_patience,
        device=device,
        warmup_epochs=args.warmup_epochs,
        lr0=args.lr0,
        momentum=args.momentum,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        
        # Augmentation
        degrees=args.rotate_limit if args.enable_augmentation else 0.0,
        fliplr=args.flip_prob if args.enable_augmentation else 0.0,
        mosaic=args.mosaic if args.enable_augmentation else 0.0,
        mixup=args.mixup if args.enable_augmentation else 0.0,
        hsv_h=args.hsv_h if args.enable_augmentation else 0.0,
        hsv_s=args.hsv_s if args.enable_augmentation else 0.0,
        hsv_v=args.hsv_v if args.enable_augmentation else 0.0
    )
    
    if has_test_split:
        print("\n--- Evaluating on Test Set ---")
        test_metrics = model.val(
            data=data_yaml_path,
            split="test" 
        )
        json_metrics_callback.log_test_metrics(test_metrics)
    
    id2label = {i: name for i, name in enumerate(class_names)}
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({"id2label": id2label}, f)
        
    print(f"Saved id2label config to {config_path}")
    print("\n--- YOLO Training and Evaluation Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a YOLO model.")
    
    # --- Args from our standard pipeline ---
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the processed Arrow dataset directory.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="YOLO model checkpoint name (e.g., yolo11m.pt).")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--model_output_root", type=str, required=True)
    parser.add_argument("--metrics_filename", type=str, required=True)
    
    # --- Task-specific Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--max_image_size", type=int, default=960)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    
    # Training configuration
    parser.add_argument("--warmup_epochs", type=float, default=3.0)
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--optimizer", type=str, default='auto', help="Optimizer (e.g., 'SGD', 'Adam', 'auto')")
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    
    # --- Augmentation Args ---
    parser.add_argument("--enable_augmentation", action="store_true")
    parser.add_argument("--flip_prob", type=float, default=0.5)
    parser.add_argument("--rotate_limit", type=float, default=0.0) # YOLO 'degrees'
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--hsv_h", type=float, default=0.015)
    parser.add_argument("--hsv_s", type=float, default=0.7)
    parser.add_argument("--hsv_v", type=float, default=0.4)
    # Brightness/Contrast generic args (ignored for YOLO in favor of HSV)
    parser.add_argument("--brightness", type=float, default=0.2)
    parser.add_argument("--contrast", type=float, default=0.2)
    
    # Accept output_dir from API
    parser.add_argument("--output_dir", type=str, default=None)

    # --- Use parse_known_args to ignore unused args passed by the generic API router ---
    args, unknown = parser.parse_known_args()
    
    if unknown:
        print(f"Warning: The following arguments were ignored by the YOLO script: {unknown}")
    
    # Construct output_dir if not passed (though API passes it)
    if not args.output_dir:
        args.output_dir = os.path.join(
            args.model_output_root, 
            f"{args.model_checkpoint.replace('.pt', '')}-{args.run_name}-{args.version}"
        )
        
    main(args)