# object_detection_utils/metrics.py

import torch
import numpy as np
from dataclasses import dataclass
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers.image_transforms import center_to_corners_format
from transformers.trainer import EvalPrediction
from typing import Mapping, Optional
from transformers import AutoImageProcessor

@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor

def convert_bbox_yolo_to_pascal(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """
    Converts bounding boxes from YOLO format to Pascal VOC format.
    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)
    # convert to absolute coordinates
    height, width = image_size
    boxes = boxes * torch.tensor([[width, height, width, height]])
    return boxes

@torch.no_grad()
def compute_metrics(
    evaluation_results: EvalPrediction,
    image_processor: AutoImageProcessor,
    threshold: float = 0.0,
    id2label: Optional[Mapping[int, str]] = None,
) -> Mapping[str, float]:
    """
    Compute mAP, mAR, and their variants, safely skipping invalid data.
    """
    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []
    valid_pred_indices = [] # Keep track of which predictions are valid

    # Collect valid targets and their original sizes
    for i, batch_targets in enumerate(targets):
        batch_valid_indices = [k for k, target in enumerate(batch_targets) if len(target["orig_size"]) == 2]
        valid_pred_indices.append(batch_valid_indices)

        for k in batch_valid_indices:
            image_target = batch_targets[k]
            boxes = torch.tensor(image_target["boxes"])
            boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
            labels = torch.tensor(image_target["class_labels"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})
            image_sizes.append(image_target["orig_size"])

    # Process only the predictions that correspond to valid targets
    for i, batch_preds in enumerate(predictions):
        valid_indices = valid_pred_indices[i]
        if not valid_indices:
            continue
            
        # Filter predictions based on valid targets from the same batch
        batch_logits = torch.tensor(batch_preds[1][valid_indices])
        batch_boxes = torch.tensor(batch_preds[2][valid_indices])
        
        # Get corresponding target sizes
        start_index = sum(len(v) for v in valid_pred_indices[:i])
        end_index = start_index + len(valid_indices)
        target_sizes = torch.tensor(image_sizes[start_index:end_index])
        
        output = ModelOutput(logits=batch_logits, pred_boxes=batch_boxes)
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

    # Compute metrics
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    # Format metrics for logging
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")

    if isinstance(classes, torch.Tensor) and classes.dim() == 0: classes = [classes]
    if isinstance(map_per_class, torch.Tensor) and map_per_class.dim() == 0: map_per_class = [map_per_class]
    if isinstance(mar_100_per_class, torch.Tensor) and mar_100_per_class.dim() == 0: mar_100_per_class = [mar_100_per_class]

    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        class_name = id2label.get(class_id.item(), f"class_{class_id.item()}")
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    return {k: round(v.item(), 4) for k, v in metrics.items()}
