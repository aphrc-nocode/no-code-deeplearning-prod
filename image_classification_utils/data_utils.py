# image_classification_utils/data_utils.py

import torch
import numpy as np
import math
import random
from datasets import Dataset
from transformers import AutoImageProcessor
from typing import List, Dict, Any

def apply_transforms(examples: Dict[str, Any], transform: Any) -> Dict[str, Any]:
    """
    Applies albumentations augmentations to a batch of images.
    This function is designed to be used with dataset.map()
    """
    # Albumentations expects a list of numpy arrays
    images = [transform(image=np.array(image.convert("RGB")))['image'] for image in examples["image"]]
    
    # Update the "image" key with augmented versions
    examples["image"] = images
    
    # Return the updated dictionary (preserving labels and other keys)
    return examples


def collate_fn(features: List[Dict[str, Any]], image_processor: AutoImageProcessor) -> Dict[str, torch.Tensor]:
    """
    Custom collate function that applies the Hugging Face image_processor
    on-the-fly to a batch of pre-augmented images.
    """
    # Extract images and labels from the list of features
    images = [feature["image"] for feature in features]
    labels = [feature["label"] for feature in features]

    # Apply the image_processor (normalization, tensor conversion)
    # This is highly optimized.
    batch = image_processor(images, return_tensors="pt")
    
    # Add the labels to the batch
    batch["labels"] = torch.tensor(labels, dtype=torch.long)
    
    return batch


def random_split(dataset: Dataset, train_ratio: float, dev_ratio: float, random_seed: int):
    """Randomly splits a single dataset into train, validation, and test sets."""
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Input must be a datasets.Dataset, but got {type(dataset)}")

    test_ratio = 1.0 - train_ratio - dev_ratio
    if not math.isclose(train_ratio + dev_ratio + test_ratio, 1.0):
        print(f"Warning: Ratios sum to {train_ratio + dev_ratio + test_ratio}, not 1.0. The remainder will form the test set.")
    
    indices = list(range(len(dataset)))
    random.seed(random_seed)
    random.shuffle(indices)

    num_items = len(indices)
    num_train = math.floor(num_items * train_ratio)
    num_dev = math.floor(num_items * dev_ratio)

    train_ds = dataset.select(indices[:num_train])
    dev_ds = dataset.select(indices[num_train : num_train + num_dev])
    test_ds = dataset.select(indices[num_train + num_dev :])

    return train_ds, dev_ds, test_ds