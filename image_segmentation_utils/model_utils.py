# image_segmentation_utils/model_utils.py

import segmentation_models_pytorch as smp
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor

def load_model(model_checkpoint, id2label, label2id):
    """
    Loads the image segmentation model with a custom classification head.
    Supports Hugging Face Transformers and Segmentation Models PyTorch (SMP).
    """
    num_labels = len(id2label)
    
    # --- Check for SMP Models (U-Net) ---
    if model_checkpoint.startswith("unet"):
        # Format: unet-resnet34 or unetplusplus-efficientnet-b0
        parts = model_checkpoint.split("-")
        arch = parts[0] # unet or unetplusplus
        encoder = "-".join(parts[1:]) # resnet34, etc.
        
        print(f"Loading SMP Model: Architecture={arch}, Encoder={encoder}, Classes={num_labels}")
        
        if arch == "unet":
            model = smp.Unet(
                encoder_name=encoder, 
                encoder_weights="imagenet", 
                in_channels=3, 
                classes=num_labels
            )
        elif arch == "unetplusplus":
            model = smp.UnetPlusPlus(
                encoder_name=encoder, 
                encoder_weights="imagenet", 
                in_channels=3, 
                classes=num_labels
            )
        else:
            raise ValueError(f"Unknown SMP architecture: {arch}")
            
        return model

    # --- Default: Hugging Face Transformers ---
    print(f"Loading Hugging Face Model: {model_checkpoint}")
    model = AutoModelForSemanticSegmentation.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id,
        num_labels=num_labels,
        ignore_mismatched_sizes=True, 
    )

    return model

def load_image_processor(model_checkpoint, max_image_size=512):
    """
    Loads the image processor.
    For SMP models, strictly speaking we don't need a HF processor, but we return 
    a dummy or a default SegFormer processor to keep the pipeline compatible if needed.
    However, our 'collate_fn' update removed the processor dependency, so this is mostly for
    compatibility with 'load_from_cache_file' logic or legacy steps.
    """
    
    if model_checkpoint.startswith("unet"):
        # We don't use a HuggingFace processor for SMP.
        # Returning None prevents double-normalization logic downstream if a user tries
        # to inject it back into inference manually before checking!
        return None
    
    processor = AutoImageProcessor.from_pretrained(
        model_checkpoint,
        do_resize=True,
        size={"height": max_image_size, "width": max_image_size},
        do_normalize=True,
        use_fast=True,
    )
    
    return processor