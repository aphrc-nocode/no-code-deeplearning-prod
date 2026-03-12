# image_segmentation_utils/data_utils.py

import torch
import numpy as np
from datasets import Dataset, DatasetDict
import math
import random
from transformers import AutoImageProcessor
from typing import List, Dict, Any

def apply_transforms(examples: Dict[str, Any], transform: Any) -> Dict[str, Any]:
    """
    Applies albumentations augmentations to a batch of images and masks.
    This function is designed to be used with dataset.map()
    
    Returns a dict with 'image' and 'mask' keys.
    """
    images = []
    masks = []
    
    for img, msk in zip(examples["image"], examples["label"]):
        img = img.convert("RGB")
        msk = msk.convert("L") # Ensure mask is single-channel
        
        transformed = transform(image=np.array(img), mask=np.array(msk))
        images.append(transformed["image"])
        masks.append(transformed["mask"])

    # This will be cached by .map()
    return {"image": images, "mask": masks}

def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function that stacks pre-augmented tensors.
    Does NOT use image_processor to avoid double-normalization.
    """
    # Extract images and masks
    # images are already tensors from ToTensorV2 in apply_transforms
    images = [feature["image"] for feature in features]
    masks = [feature["mask"] for feature in features]

    batch = {}
    # Stack the list of 3D tensors (C, H, W) into a 4D batch tensor (B, C, H, W)
    batch["pixel_values"] = torch.stack([torch.tensor(img) if not isinstance(img, torch.Tensor) else img for img in images])
    
    # Stack masks and cast to long for segmentation labels
    batch["labels"] = torch.stack([torch.tensor(m) if not isinstance(m, torch.Tensor) else m for m in masks]).long()
    
    return batch

def random_split(dataset: Dataset, train_ratio: float, dev_ratio: float, random_seed: int):
    """
    Randomly splits a single dataset into train, validation, and test sets.
    This is identical to the image classification util.
    """
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