# model_registry.py
# This file is the single source of truth for all supported models.
# The backend API will read from this file.

MODEL_REGISTRY = {

    "object_detection": {
        "DETR": [
            "facebook/detr-resnet-50",
            "facebook/detr-resnet-101"
        ],
        "YOLOS": [
            "hustvl/yolos-tiny",
            "hustvl/yolos-small",
            "hustvl/yolos-base"
        ],
        "YOLO": [
            "yolo11n.pt",
            "yolo11s.pt",
            "yolo11m.pt",
            "yolo11l.pt",
            # --- YOLOv12 (Attention-Centric 2025) ---
            "yolov12n.pt",
            "yolov12s.pt",
            "yolov12m.pt",
            "yolov12l.pt",
            "yolov12x.pt",
            # --- YOLO26 (Latest SOTA 2026) ---
            "yolo26n.pt",
            "yolo26s.pt",
            "yolo26m.pt",
            "yolo26l.pt",
            "yolo26x.pt"
        ],
        "RT-DETR": [
            "rtdetr-l.pt",
            "rtdetr-x.pt"
        ]
    },
    "image_classification": {
        "ViT": [
            "google/vit-base-patch16-224-in21k",
            "google/vit-large-patch16-224-in21k"
        ],
        "BEiT": [
            "microsoft/beit-base-patch16-224-pt22k",
            "microsoft/beit-large-patch16-224-pt22k"
        ],
        "Swin": [
            "microsoft/swin-base-patch4-window7-224",
            "microsoft/swin-tiny-patch4-window7-224"
        ],
        "Swin V2": [
             "microsoft/swinv2-base-patch4-window12-192-22k"
        ],
        "ConvNeXt": [
            "facebook/convnext-tiny-224",
            "facebook/convnext-small-224",
            "facebook/convnext-base-224"
        ],
        "EfficientNet": [
             "google/efficientnet-b0",
             "google/efficientnet-b1",
             "google/efficientnet-b2",
             "google/efficientnet-b4"
        ],
        "ResNet": [
             "microsoft/resnet-50",
             "microsoft/resnet-101"
        ],
        "DINOv2 (ViT)": [
             "facebook/dinov2-base"
        ]
    },
    "semantic_segmentation": {
        "SegFormer": [
            "nvidia/segformer-b0-finetuned-ade-512-512",
            "nvidia/segformer-b1-finetuned-ade-512-512",
            "nvidia/segformer-b2-finetuned-ade-512-512"
        ],
        "U-Net (Medical SOTA)": [
            "unet-resnet18",
            "unet-resnet34",
            "unet-resnet50",
            "unet-efficientnet-b0"
        ],
        "U-Net++": [
            "unetplusplus-resnet34",
            "unetplusplus-efficientnet-b0"
        ]
    }
}

def get_registry():
    """Returns the full model registry."""
    return MODEL_REGISTRY

def get_task_models(task_type: str):
    """Returns the model registry for a specific task."""
    return MODEL_REGISTRY.get(task_type, {})

if __name__ == "__main__":
    import json
    print(json.dumps(MODEL_REGISTRY, indent=2))