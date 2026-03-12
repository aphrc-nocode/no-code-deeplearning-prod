# object_detection_inference_yolos.py

import argparse
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import os

def run_yolos_inference(model_checkpoint: str, image_path: str, output_path: str, threshold: float = 0.9):
    """
    Runs inference on a single image using a YOLOS model,
    which requires special post-processing.
    """
    print(f"Loading YOLOS processor and model from: {model_checkpoint}")
    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
    model = AutoModelForObjectDetection.from_pretrained(model_checkpoint)

    image = Image.open(image_path).convert("RGB")
    
    print(f"Running YOLOS inference with threshold: {threshold}")
    
    # 1. Pre-process the image
    inputs = image_processor(images=image, return_tensors="pt")

    # 2. Run the model
    with torch.no_grad():
        outputs = model(**inputs)

    # 3. Post-process (The YOLOS-specific logic)
    # Get probabilities and exclude the "no-object" class
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    
    # Apply the confidence threshold
    keep = probas.max(-1).values > threshold
    
    # Get scaled bounding boxes
    target_sizes = torch.tensor([image.size[::-1]])
    # We must use .post_process(), NOT .post_process_object_detection()
    postprocessed_outputs = image_processor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes']

    # 4. Draw the results
    draw = ImageDraw.Draw(image)
    
    print(f"Found {keep.sum()} objects with confidence > {threshold}")
    
    # Filter using the 'keep' mask and draw
    for prob, box in zip(probas[keep], bboxes_scaled[keep]):
        box = [round(i, 2) for i in box.tolist()]
        
        # Get class and score
        cl = prob.argmax()
        score = prob[cl]
        label_text = model.config.id2label[cl.item()]
        
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
    parser = argparse.ArgumentParser(description="Run YOLOS object detection inference.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the trained model checkpoint directory.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output image.")
    parser.add_argument("--threshold", type=float, default=0.9, help="Confidence threshold for predictions.")
    
    args = parser.parse_args()
    
    run_yolos_inference(
        model_checkpoint=args.model_checkpoint,
        image_path=args.image_path,
        output_path=args.output_path,
        threshold=args.threshold
    )