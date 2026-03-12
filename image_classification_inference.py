# image_classification_inference.py

import argparse
from transformers import pipeline
from PIL import Image

def main(args):
    """Runs image classification inference and prints the result."""
    import torch
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'cuda:0' if device == 0 else 'cpu'}")

    classifier = pipeline("image-classification", model=args.model_checkpoint, device=device)
    image = Image.open(args.image_path)
    results = classifier(image)
    
    top_prediction = results[0]
    print(f"Predicted class: {top_prediction['label']} (Confidence: {top_prediction['score']:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run image classification inference.")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()
    main(args)