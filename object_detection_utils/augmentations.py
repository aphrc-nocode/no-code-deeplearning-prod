# object_detection_utils/augmentations.py

import albumentations as A

def get_train_transform(max_image_size=600, config=None):
    """
    Returns the training augmentation pipeline.
    
    Args:
        max_image_size (int): Max size for resizing.
        config (dict, optional): Dictionary with augmentation params:
            - enable_augmentation (bool)
            - flip_prob (float)
            - rotate_limit (float)
            - brightness (float)
            - contrast (float)
    """
    if config is None:
        config = {}
        
    transforms = []
    
    # Always include resize/crop logic? Or does that depend on config?
    # Usually we always want some resizing for batching, but SafeCrop is an augmentation.
    # The original code had SmallestMaxSize + RandomSizedBBoxSafeCrop.
    # Let's keep basic resizing active but make heavy augs conditional.
    
    # 1. Base Resizing (Always active for consistency, or maybe just SmallestMaxSize?)
    # If we disable augmentation, we probably just want resizing.
    if config.get("enable_augmentation", False):
         transforms.append(
            A.OneOf([
                A.SmallestMaxSize(max_size=max_image_size, p=1.0),
                # A.RandomSizedBBoxSafeCrop(height=max_image_size, width=max_image_size, p=1.0) # This can be aggressive
            ], p=1.0) # Ensure size consistency? 
         )
         # Actually, RandomSizedBBoxSafeCrop forces a specific size. SmallestMaxSize just scales.
         # For DETR, we often just want efficient resizing.
         transforms.append(A.SmallestMaxSize(max_size=max_image_size, p=1.0))
         
         # 2. Geometric Augmentations
         if config.get("flip_prob", 0.0) > 0:
             transforms.append(A.HorizontalFlip(p=config.get("flip_prob", 0.5)))
             
         if config.get("rotate_limit", 0) > 0:
             transforms.append(A.SafeRotate(limit=config.get("rotate_limit"), p=0.5))
             
         # 3. Color Augmentations
         brightness = config.get("brightness", 0.0)
         contrast = config.get("contrast", 0.0)
         if brightness > 0 or contrast > 0:
             transforms.append(A.RandomBrightnessContrast(
                 brightness_limit=brightness, 
                 contrast_limit=contrast, 
                 p=0.5
             ))
             
         # 4. Medical-Grade Deformations (Grid/Elastic)
         # Helpful for tissue/organ variability
         if config.get("enable_augmentation", False):
             transforms.append(
                A.OneOf([
                    A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                    A.ElasticTransform(alpha=1, sigma=50, p=1.0)
                ], p=0.3)
             )
    else:
         # Minimal transform (resize only)
         transforms.append(A.SmallestMaxSize(max_size=max_image_size, p=1.0))
    
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, filter_invalid_bboxes=True, min_visibility=0.5),
    )

def get_validation_transform():
    """Returns the validation transformation pipeline (no actual augmentation)."""
    return A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
    )
