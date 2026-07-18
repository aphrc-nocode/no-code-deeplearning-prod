# image_classification_explain.py
#
# Grad-CAM explainability for image-classification models. Produces a heatmap
# overlay showing which regions of the image drove the predicted class, so a
# no-code user can see *why* a prediction was made — not just the label.
#
# Works across the registry's two model families:
#   - CNNs (ResNet, ConvNeXt, EfficientNet): target the last Conv2d layer.
#   - Transformers (ViT, BEiT, Swin, DINOv2): target the final LayerNorm and
#     reshape the token sequence back to a 2D grid.

import argparse
import json

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class HFClassifierWrapper(torch.nn.Module):
    """Return raw logits (a plain tensor) so Grad-CAM can call .backward()."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(pixel_values=x).logits


def _grid_reshape(tensor):
    """Reshape a (B, tokens, C) transformer output to (B, C, H, W).

    Drops the leading CLS token when the token count is a perfect square + 1;
    infers a square grid from the number of patch tokens.
    """
    n_tokens = tensor.size(1)
    n_patches = n_tokens - 1 if int((n_tokens - 1) ** 0.5) ** 2 == (n_tokens - 1) else n_tokens
    start = n_tokens - n_patches
    side = int(round(n_patches ** 0.5))
    result = tensor[:, start:start + side * side, :].reshape(
        tensor.size(0), side, side, tensor.size(2)
    )
    return result.permute(0, 3, 1, 2)


def select_target_layer(model):
    """Pick a Grad-CAM target layer and (optionally) a reshape transform.

    Returns (target_layers, reshape_transform_or_None).
    """
    conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    if conv_layers:
        # CNN: the last conv layer holds the richest localized features.
        return [conv_layers[-1]], None

    # Transformer: use the last LayerNorm and reshape tokens to a grid.
    norm_layers = [m for m in model.modules() if isinstance(m, torch.nn.LayerNorm)]
    if not norm_layers:
        raise ValueError("Could not find a Conv2d or LayerNorm target layer for Grad-CAM.")
    return [norm_layers[-1]], _grid_reshape


def run_explain(model_checkpoint: str, image_path: str, output_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = AutoImageProcessor.from_pretrained(model_checkpoint)
    model = AutoModelForImageClassification.from_pretrained(model_checkpoint).to(device).eval()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"]

    # Predicted class.
    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits
    pred_idx = int(logits.argmax(-1).item())
    pred_label = model.config.id2label.get(pred_idx, str(pred_idx))
    confidence = float(torch.softmax(logits, dim=-1)[0, pred_idx].item())
    print(f"Predicted class: {pred_label} (Confidence: {confidence:.4f})")

    # Grad-CAM.
    target_layers, reshape = select_target_layer(model)
    wrapper = HFClassifierWrapper(model)
    cam = GradCAM(model=wrapper, target_layers=target_layers, reshape_transform=reshape)
    grayscale_cam = cam(
        input_tensor=pixel_values,
        targets=[ClassifierOutputTarget(pred_idx)],
    )[0]

    # Overlay the heatmap on the (normalized-to-0..1) processed image so the CAM
    # aligns with what the model actually saw.
    vis = pixel_values[0].detach().cpu().numpy().transpose(1, 2, 0)
    vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
    overlay = show_cam_on_image(vis, grayscale_cam, use_rgb=True)
    Image.fromarray(overlay).save(output_path)
    print(f"Explanation saved to: {output_path}")

    return {"prediction": pred_label, "confidence": confidence, "output_path": output_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grad-CAM explanation for image classification.")
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    result = run_explain(args.model_checkpoint, args.image_path, args.output_path)
    # Emit a JSON line so the API can parse the prediction alongside the image.
    print("RESULT_JSON:" + json.dumps(result))
