# object_detection_inference_yolo.py

import argparse
import os
import torch
from ultralytics import YOLO
from PIL import Image

def run_yolo_inference(model_checkpoint: str, image_path: str, output_path: str, threshold: float, iou: float, max_det: int, imgsz: int = 640, classes=None):
    """
    Runs YOLO inference on a single image and saves the annotated result.
    
    --- MODIFIED (V3) ---
    This version uses save=False and results[0].plot() to get the
    annotated result. Added imgsz and classes support.
    """
    print(f"Loading YOLO model from: {model_checkpoint}")
    
    if not model_checkpoint.endswith(".pt"):
        model_path = os.path.join(model_checkpoint, "weights", "best.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find 'best.pt' in {model_checkpoint}")
    else:
        model_path = model_checkpoint
        
    model = YOLO(model_path)
    device = 0 if torch.cuda.is_available() else "cpu"
    
    print(f"Running YOLO inference on {image_path} with conf={threshold}, iou={iou}, imgsz={imgsz}, classes={classes}")
    

    # 1. Run prediction with save=False
    results = model.predict(
        source=image_path,
        conf=threshold,
        iou=iou,
        max_det=max_det,
        imgsz=imgsz,
        classes=classes,
        device=device,
        save=False # Do not save to disk
    )
    
    # 2. Get the annotated image from the results object
    if results and len(results) > 0:
        # results[0].plot() returns the annotated image as a BGR NumPy array
        annotated_array_bgr = results[0].plot()
        
        # 3. Convert from BGR (OpenCV format) to RGB (PIL format)
        annotated_array_rgb = annotated_array_bgr[..., ::-1]
        
        # 4. Save the image using PIL
        annotated_image = Image.fromarray(annotated_array_rgb)
        annotated_image.save(output_path)
        print(f"Output image saved to: {output_path}")
    else:
        raise Exception("YOLO inference produced no results.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO object detection inference.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the trained model (directory or .pt file).")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output annotated image.")
    parser.add_argument("--threshold", type=float, default=0.25, help="Confidence threshold for predictions.")
    
    # Ensure new args are parsed
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS.")
    parser.add_argument("--max_det", type=int, default=300, help="Maximum number of detections.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--classes", type=str, default=None, help="Comma-separated class IDs (e.g. '0,2').")
    
    args = parser.parse_args()

    # Parse classes string to list of ints
    if args.classes:
        try:
            # Handle "0, 2" or "0" or "0,2,"
            args.classes = [int(c.strip()) for c in args.classes.split(",") if c.strip()]
        except ValueError:
             print(f"Warning: Invalid classes format '{args.classes}'. Ignoring filter.")
             args.classes = None
    
    run_yolo_inference(
        model_checkpoint=args.model_checkpoint,
        image_path=args.image_path,
        output_path=args.output_path,
        threshold=args.threshold,
        iou=args.iou,
        max_det=args.max_det,
        imgsz=args.imgsz,
        classes=args.classes
    )