# image_segmentation_inference.py

import argparse
import torch
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
from PIL import Image
import numpy as np
import os

def run_inference(model_checkpoint: str, image_path: str, output_path: str):
    """
    Runs inference on a single image using a fine-tuned segmentation model
    and saves the resulting mask overlay.
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check if this is an SMP pipeline by looking at the config
    config_path = os.path.join(model_checkpoint, "config.json")
    is_smp = False
    if os.path.exists(config_path):
        import json
        with open(config_path, "r") as f:
            config = json.load(f)
            if config.get("architectures", [None])[0] in ["Unet", "UnetPlusPlus"]:
                is_smp = True
    
    # 1. Load Processor / Transform and Model
    print(f"Loading model from: {model_checkpoint}")
    
    if is_smp:
        import segmentation_models_pytorch as smp
        from image_segmentation_utils.augmentations import get_validation_transform
        
        # Determine architecture
        arch = config.get("architectures")[0]
        encoder_name = config.get("encoder_name", "resnet34") # Fallback
        
        # Load ID to Label
        id2label = {int(k): v for k, v in config.get("id2label", {}).items()}
        num_classes = len(id2label) if id2label else config.get("num_labels", 2)
        
        if arch == "Unet":
            model = smp.Unet(encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=num_classes)
        else:
            model = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=num_classes)
            
        model.load_state_dict(torch.load(os.path.join(model_checkpoint, "pytorch_model.bin"), map_location=device))
        
        # SMP needs albumentations validation transform
        # SMP inference requires a predefined size, assuming 512 if not specified.
        transform = get_validation_transform(max_image_size=config.get("image_size", 512))
        
        image = Image.open(image_path).convert("RGB")
        inputs = {"pixel_values": transform(image=np.array(image))["image"].unsqueeze(0).to(device)}
        
    else:
        # Standard Hugging Face loading
        image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
        model = AutoModelForSemanticSegmentation.from_pretrained(model_checkpoint)
        
        image = Image.open(image_path).convert("RGB")
        inputs = image_processor(images=image, return_tensors="pt").to(device)

    model.to(device)
    model.eval()

    # 2. Run inference
    with torch.no_grad():
        if is_smp:
            outputs = model(inputs["pixel_values"])
            logits = outputs # SMP returns logits directly
        else:
            outputs = model(**inputs)
            logits = outputs.logits
    
    # 3. Post-process the logits

    # 4. Create a color mask
    # We need the color map. Let's try to get it from the model config.
    color_map = None
    if hasattr(model.config, "id2label"):
        id2label = model.config.id2label
        # Create a simple color map based on label ID
        # We'll just generate random colors for this demo
        palette = np.random.randint(0, 255, (len(id2label), 3), dtype=np.uint8)
        palette[0] = [0, 0, 0] # Make background (label 0) black
        
        color_map = palette[pred_seg]
        color_mask = Image.fromarray(color_map.astype(np.uint8), "RGB")

    # 5. Overlay the mask on the original image
    if color_map is not None:
        overlay_image = Image.blend(image, color_mask, alpha=0.5)
    else:
        # Fallback: just save the raw segmentation map (will look black/white)
        overlay_image = Image.fromarray(pred_seg.astype(np.uint8))

    # 6. Save the final image
    overlay_image.save(output_path)
    print(f"Output image saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run image segmentation inference.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the trained model checkpoint directory.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output segmentation overlay.")
    
    args = parser.parse_args()
    
    run_inference(
        model_checkpoint=args.model_checkpoint,
        image_path=args.image_path,
        output_path=args.output_path
    )