# object_detection_utils/model_utils.py

import torch
from transformers import AutoConfig, AutoModelForObjectDetection, AutoImageProcessor

def load_model(model_checkpoint, id2label, label2id):
    """Loads the object detection model using a custom config."""
    config = AutoConfig.from_pretrained(
        model_checkpoint,
        label2id=label2id,
        id2label=id2label,
    )
    model = AutoModelForObjectDetection.from_pretrained(
        model_checkpoint,
        config=config,
        ignore_mismatched_sizes=True,
    )
    return model

def load_image_processor(model_checkpoint, max_image_size):
    """Loads the image processor with specific resizing and padding."""
    return AutoImageProcessor.from_pretrained(
        model_checkpoint,
        do_resize=True,
        size={"max_height": max_image_size, "max_width": max_image_size},
        do_pad=True,
        pad_size={"height": max_image_size, "width": max_image_size},
        use_fast=True,
    )

def collate_fn(batch):
    """Custom collate function for object detection."""
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # The image_processor handles padding, so we can just stack pixel_values
    encoded_batch = {
        "pixel_values": torch.stack(pixel_values),
        "labels": labels
    }
    return encoded_batch
