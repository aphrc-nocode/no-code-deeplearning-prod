# image_segmentation_utils/metrics.py

import evaluate
import numpy as np
from transformers import EvalPrediction
from typing import Callable, Dict, Any

def create_compute_metrics_fn(num_labels: int, id2label: Dict[int, str], ignore_index: int = 255) -> Callable:
    """
    Factory function to create the compute_metrics function
    with baked-in num_labels and id2label.
    """
    
    metric = evaluate.load("mean_iou")

    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Computes Mean IoU, Mean Accuracy, and per-class IoU
        for a semantic segmentation model.
        """
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        
        # Models like SegFormer output a tuple.
        # The first element [0] is the segmentation map logits.
        if isinstance(logits, tuple):
            logits = logits[0]
            
        # (batch_size, num_classes, height, width) -> (batch_size, height, width)
        predictions = np.argmax(logits, axis=1)

        # Flatten to (batch_size * height * width)
        pred_flat = predictions.flatten()
        label_flat = labels.flatten()

        try:
            metrics = metric.compute(
                predictions=pred_flat,
                references=label_flat,
                num_labels=num_labels,
                ignore_index=ignore_index,
                # reduce_labels=False is critical for handling batches
                # that don't contain every single class.
                reduce_labels=False, 
            )
            
            # Format metrics
            final_metrics = {}
            
            # Main metrics
            final_metrics["eval_mean_iou"] = round(metrics["mean_iou"], 4)
            final_metrics["eval_mean_accuracy"] = round(metrics["mean_accuracy"], 4)

            # Add per-class IoU
            if "per_category_iou" in metrics:
                for i, iou in enumerate(metrics["per_category_iou"]):
                    label_name = id2label.get(i, f"class_{i}")
                    # Prepend "iou_" to avoid collisions with "loss", etc.
                    final_metrics[f"eval_iou_{label_name}"] = round(iou, 4)
            
            return final_metrics

        except Exception as e:
            print(f"Error during metric computation: {e}")
            return {"eval_mean_iou": 0.0, "eval_mean_accuracy": 0.0}

    return compute_metrics