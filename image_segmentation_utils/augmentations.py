# image_segmentation_utils/augmentations.py

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(
    max_image_size=512,
    enable_augmentation=False,
    flip_prob=0.5,
    rotate_limit=15,
    brightness=0.2,
    contrast=0.2
):
    """
    Returns the training augmentation pipeline for semantic segmentation.
    Augmentation is optional and configurable.
    """
    
    # 1. Base Transforms (Always Applied)
    transforms = [
        # Resize/crop to a fixed size.
        A.RandomResizedCrop(
            size=(max_image_size, max_image_size),
            scale=(0.5, 1.0), 
            p=1.0
        )
    ]
    
    # 2. Add Conditional Augmentations
    if enable_augmentation:
        if flip_prob > 0:
            transforms.append(A.HorizontalFlip(p=flip_prob))
        
        if rotate_limit > 0:
            transforms.append(A.Rotate(limit=rotate_limit, p=0.5))
            
        # Color augmentations (only applied to image)
        if brightness > 0 or contrast > 0:
            transforms.append(
                A.ColorJitter(
                    brightness=brightness, 
                    contrast=contrast, 
                    saturation=0.2, # keep default
                    hue=0.1, # keep default 
                    p=0.8
                )
            )

            
        # 3. Medical-Grade Deformations (Grid/Elastic) & Cutout
        transforms.append(
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, p=1.0)
            ], p=0.3)
        )
        
        transforms.append(
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, min_height=8, min_width=8, p=0.2)
        )

    # 3. Normalization & Conversion (Always Applied)
    transforms.extend([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(), # Converts image and mask to PyTorch tensors
    ])

    return A.Compose(transforms)

def get_validation_transform(max_image_size=512):
    """
    Returns the validation transformation pipeline.
    Just resizes, normalizes, and converts to tensor.
    """
    return A.Compose([
        A.Resize(height=max_image_size, width=max_image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])