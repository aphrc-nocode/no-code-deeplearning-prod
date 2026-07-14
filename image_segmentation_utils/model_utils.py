# image_segmentation_utils/model_utils.py

import segmentation_models_pytorch as smp
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor, PretrainedConfig

# Maps our SMP architecture slug to the class name we persist in config.json and
# read back at inference time.
_SMP_ARCH_TO_CLASSNAME = {"unet": "Unet", "unetplusplus": "UnetPlusPlus"}


def _attach_hf_config(model, arch, encoder, id2label, label2id, num_labels):
    """
    Give an SMP model a Hugging Face PretrainedConfig.

    The HF Trainer assigns `model.config.use_cache = ...` in its constructor, but
    SMP (>=0.5) exposes `.config` as a read-only property returning a plain dict.
    We shadow that property with a per-instance subclass so `.config` becomes a
    normal, settable attribute, then point it at a PretrainedConfig that both
    satisfies the Trainer and carries the metadata needed to rebuild the model
    at inference time.
    """
    config = PretrainedConfig()
    config.architectures = [_SMP_ARCH_TO_CLASSNAME.get(arch, arch)]
    config.encoder_name = encoder
    config.num_labels = num_labels
    config.id2label = {int(k): v for k, v in id2label.items()}
    config.label2id = {k: int(v) for k, v in label2id.items()}
    config.use_cache = False

    # SMP's load_state_dict(self, state_dict, **kwargs) accepts `strict` only as a
    # keyword, but the HF Trainer (load_best_model_at_end) passes it positionally.
    # Adapt the positional call to a keyword one, preserving SMP's key remapping.
    base_cls = model.__class__

    def _load_state_dict(self, state_dict, strict=True, *args, **kwargs):
        return base_cls.load_state_dict(self, state_dict, strict=strict, **kwargs)

    # Shadow the read-only `config` property with a plain class attribute (so the
    # Trainer can assign model.config.use_cache) and install the load_state_dict
    # adapter, both on a throwaway per-instance subclass. SMP's class is untouched.
    model.__class__ = type(
        f"{base_cls.__name__}HF",
        (base_cls,),
        {"config": None, "load_state_dict": _load_state_dict},
    )
    model.config = config
    return model


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

        return _attach_hf_config(model, arch, encoder, id2label, label2id, num_labels)

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