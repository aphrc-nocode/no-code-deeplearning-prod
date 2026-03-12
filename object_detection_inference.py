# object_detection_inference.py

import argparse
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import os

def run_inference(model_checkpoint: str, image_path: str, output_path: str, threshold: float = 0.5):
    """
    Runs inference on a single image using a fine-tuned object detection model.
    """
    # Load model and processor from the same checkpoint directory
    print(f"Loading processor and model from: {model_checkpoint}")
    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
    model = AutoModelForObjectDetection.from_pretrained(model_checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)

    image = Image.open(image_path).convert("RGB")
    
    print(f"Running inference with threshold: {threshold}")
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

    draw = ImageDraw.Draw(image)
    
    print(f"Found {len(results['scores'])} objects with confidence > {threshold}")
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        label_text = model.config.id2label[label.item()]
        
        draw.rectangle(box, outline="red", width=3)
        
        label_str = f"{label_text}: {round(score.item(), 2)}"
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
            
        text_bbox = draw.textbbox((box[0], box[1]), label_str, font=font)
        text_height = text_bbox[3] - text_bbox[1]
        text_y_position = box[1] - text_height - 5 if box[1] > text_height + 5 else box[1]

        draw.rectangle((box[0], text_y_position, box[0] + text_bbox[2] - text_bbox[0], text_y_position + text_height), fill="red")
        draw.text((box[0], text_y_position), label_str, fill="white", font=font)

    image.save(output_path)
    print(f"Output image saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run object detection inference.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the trained model checkpoint directory.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output image.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for predictions.")
    
    args = parser.parse_args()
    
    run_inference(
        model_checkpoint=args.model_checkpoint,
        image_path=args.image_path,
        output_path=args.output_path,
        threshold=args.threshold
    )