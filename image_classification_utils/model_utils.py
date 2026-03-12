# image_classification_utils/model_utils.py

from transformers import AutoModelForImageClassification, AutoImageProcessor

def load_model(model_checkpoint, id2label, label2id, labels):
    """Loads the image classification model."""
    
    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint,
        label2id=label2id,
        id2label=id2label,
        num_labels=len(labels),
        ignore_mismatched_sizes=True,
    )

    return model

def load_image_processor(model_checkpoint, max_image_size):
    """Loads the image processor with specific resizing."""
    # Load first with defaults to detect the required size keys (e.g., shortest_edge vs height/width)
    # We try to use the fast implementation now that we are providing the correct size config
    processor = AutoImageProcessor.from_pretrained(model_checkpoint, use_fast=True)
    
    # Check default size keys
    default_size = getattr(processor, "size", {})
    size_config = {"height": max_image_size, "width": max_image_size}
    
    # If the model expects 'shortest_edge' (like ConvNeXt), adapt accordingly
    if "shortest_edge" in default_size:
        size_config = {"shortest_edge": max_image_size}
    
    # Update the processor with our forced size
    # We reload or simply update parameters. Since we already loaded it, updating is faster.
    processor.do_resize = True
    processor.size = size_config
    
    return processor
