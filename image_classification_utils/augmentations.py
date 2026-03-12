# image_classification_utils/augmentations.py

import albumentations as A


def get_train_transform(
    max_image_size=224,
    enable_augmentation=False,
    flip_prob=0.5,
    rotate_limit=15,
    brightness=0.2,
    contrast=0.2
):
    """
    Returns the training augmentation pipeline for image classification.
    Augmentation is optional and configurable.
    """
    
    # 1. Base Transforms (Always Applied)
    # We always crop/resize to ensure consistent input size before the model
    transforms = [
        A.RandomResizedCrop(
            size=(max_image_size, max_image_size), 
            scale=(0.8, 1.0),
            p=1.0 # Always crop
        )
    ]
    
    # 2. Add Conditional Augmentations
    if enable_augmentation:
        if flip_prob > 0:
            transforms.append(A.HorizontalFlip(p=flip_prob))
        
        if rotate_limit > 0:
            transforms.append(A.Rotate(limit=rotate_limit, p=0.5))
            
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
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), p=0.2)
        )

    return A.Compose(transforms)

def get_validation_transform():
    """Returns the validation transformation pipeline"""
    return A.Compose([
        A.NoOp(),
    ])