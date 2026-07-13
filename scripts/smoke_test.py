#!/usr/bin/env python3
"""End-to-end smoke test for the deep-learning pipelines.

Generates tiny synthetic datasets and runs the real training and inference
scripts for every task and model family, asserting each completes and produces
the expected artifacts. Designed to run on CPU in CI as a regression guard: it
is what catches library-version drift before it reaches production.

Run:  python scripts/smoke_test.py
Env overrides (used by CI to pick small models):
    SMOKE_CLS_MODEL, SMOKE_DET_HF_MODEL, SMOKE_DET_YOLO_MODEL,
    SMOKE_SEG_HF_MODEL, SMOKE_SEG_SMP_MODEL
    SMOKE_KEEP=1   keep the work dir for inspection
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable

CLS_MODEL = os.environ.get("SMOKE_CLS_MODEL", "google/efficientnet-b0")
DET_HF_MODEL = os.environ.get("SMOKE_DET_HF_MODEL", "facebook/detr-resnet-50")
DET_YOLO_MODEL = os.environ.get("SMOKE_DET_YOLO_MODEL", "yolo11n.pt")
SEG_HF_MODEL = os.environ.get("SMOKE_SEG_HF_MODEL", "nvidia/segformer-b0-finetuned-ade-512-512")
SEG_SMP_MODEL = os.environ.get("SMOKE_SEG_SMP_MODEL", "unet-resnet18")

rng = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# Fixture generation (synthetic data only)
# --------------------------------------------------------------------------- #
def make_classification(root: Path):
    for split, n in {"train": 6, "validation": 2, "test": 2}.items():
        for cls, (lo, hi) in {"dark": (0, 80), "bright": (150, 256)}.items():
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                arr = rng.integers(lo, hi, (64, 64, 3), dtype=np.uint8)
                Image.fromarray(arr, "RGB").save(d / f"{cls}_{i}.png")


def make_detection(root: Path):
    import json
    for split, n in {"train": 6, "validation": 2, "test": 2}.items():
        d = root / split
        d.mkdir(parents=True, exist_ok=True)
        images, annotations, ann_id = [], [], 1
        for img_id in range(n):
            img = rng.integers(0, 60, (96, 96, 3), dtype=np.uint8)
            bw, bh = int(rng.integers(18, 30)), int(rng.integers(18, 30))
            x, y = int(rng.integers(0, 96 - bw)), int(rng.integers(0, 96 - bh))
            img[y:y + bh, x:x + bw] = rng.integers(180, 256, 3, dtype=np.uint8)
            fname = f"img_{img_id}.png"
            Image.fromarray(img, "RGB").save(d / fname)
            images.append({"id": img_id, "file_name": fname, "width": 96, "height": 96})
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                                "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})
            ann_id += 1
        (d / "_annotations.coco.json").write_text(json.dumps(
            {"images": images, "annotations": annotations,
             "categories": [{"id": 1, "name": "object", "supercategory": "none"}]}))


def make_segmentation(root: Path):
    for split, n in {"train": 6, "validation": 2, "test": 2}.items():
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "masks").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = rng.integers(0, 90, (64, 64, 3), dtype=np.uint8)
            mask = np.zeros((64, 64), dtype=np.uint8)
            y, x = int(rng.integers(8, 36)), int(rng.integers(8, 36))
            img[y:y + 18, x:x + 18] = rng.integers(170, 256, 3, dtype=np.uint8)
            mask[y:y + 18, x:x + 18] = 1
            Image.fromarray(img, "RGB").save(root / split / "images" / f"s_{i}.png")
            Image.fromarray(mask, "L").save(root / split / "masks" / f"s_{i}.png")


# --------------------------------------------------------------------------- #
# Runner helpers
# --------------------------------------------------------------------------- #
def run(cmd, env):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-3000:]); print(p.stderr[-3000:])
    return p.returncode == 0


def find_run_dir(root: Path, prefix: str):
    hits = [d for d in root.glob(f"{prefix}*") if d.is_dir()]
    return hits[0] if hits else None


# --------------------------------------------------------------------------- #
# Task smoke tests — each returns (name, ok)
# --------------------------------------------------------------------------- #
def smoke_classification(work: Path, env):
    data, out = work / "cls_data", work / "cls_out"
    make_classification(data)
    ok = run([PY, "image_classification_train.py", "--data_dir", data,
              "--model_checkpoint", CLS_MODEL, "--is_presplit", "--epochs", "1",
              "--train_batch_size", "4", "--eval_batch_size", "4", "--max_image_size", "224",
              "--num_proc", "1", "--model_output_root", out, "--metrics_filename", "m.json",
              "--run_name", "smoke", "--version", "0"], env)
    ckpt = find_run_dir(out, CLS_MODEL.split("/")[-1])
    if ok and ckpt:
        img = next((data / "test").rglob("*.png"))
        ok = run([PY, "image_classification_inference.py",
                  "--model_checkpoint", ckpt, "--image_path", img], env)
    return ("classification", ok and ckpt is not None)


def smoke_detection_hf(work: Path, env):
    data, proc, out = work / "det_data", work / "det_proc", work / "det_hf_out"
    make_detection(data)
    ok = run([PY, "object_detection_train.py", "--data_dir", data, "--processed_data_dir", proc,
              "--model_checkpoint", DET_HF_MODEL, "--epochs", "1", "--train_batch_size", "2",
              "--eval_batch_size", "2", "--max_image_size", "256", "--num_proc", "1",
              "--model_output_root", out, "--metrics_filename", "m.json",
              "--run_name", "smoke", "--version", "0"], env)
    ckpt = find_run_dir(out, DET_HF_MODEL.split("/")[-1])
    if ok and ckpt:
        img = next((data / "test").glob("*.png"))
        ok = run([PY, "object_detection_inference.py", "--model_checkpoint", ckpt,
                  "--image_path", img, "--output_path", work / "det_hf_pred.png",
                  "--threshold", "0.1"], env)
    return ("detection-DETR", ok and ckpt is not None)


def smoke_detection_yolo(work: Path, env):
    data, proc, out = work / "det_data", work / "det_proc", work / "det_yolo_out"
    if not proc.exists():          # reuse the Arrow dataset the DETR run built
        make_detection(data)
        subprocess.run([PY, "object_detection_utils/preprocess_data.py",
                        "--raw_data_dir", data, "--processed_data_dir", proc],
                       cwd=REPO, env=env, capture_output=True, text=True)
    run_dir = out / "yolo-run"
    ok = run([PY, "object_detection_train_yolo.py", "--data_dir", proc,
              "--model_checkpoint", DET_YOLO_MODEL, "--run_name", "smoke", "--version", "0",
              "--model_output_root", out, "--metrics_filename", "m.json",
              "--output_dir", run_dir, "--epochs", "2", "--train_batch_size", "2",
              "--max_image_size", "96", "--num_proc", "1", "--early_stopping_patience", "5"], env)
    best = run_dir / "weights" / "best.pt"
    if ok and best.exists():
        img = next((data / "test").glob("*.png"))
        ok = run([PY, "object_detection_inference_yolo.py", "--model_checkpoint", run_dir,
                  "--image_path", img, "--output_path", work / "det_yolo_pred.png",
                  "--threshold", "0.1", "--iou", "0.7", "--max_det", "300", "--imgsz", "96"], env)
    return ("detection-YOLO", ok and best.exists())


def smoke_segmentation(work: Path, env, model, tag):
    data, out = work / "seg_data", work / f"seg_{tag}_out"
    make_segmentation(data)
    ok = run([PY, "image_segmentation_train.py", "--data_dir", data,
              "--model_checkpoint", model, "--is_presplit", "--epochs", "1",
              "--train_batch_size", "2", "--eval_batch_size", "2", "--max_image_size", "64",
              "--num_proc", "1", "--model_output_root", out, "--metrics_filename", "m.json",
              "--run_name", tag, "--version", "0"], env)
    ckpt = find_run_dir(out, model.split("/")[-1])
    if ok and ckpt:
        img = next((data / "test" / "images").glob("*.png"))
        ok = run([PY, "image_segmentation_inference.py", "--model_checkpoint", ckpt,
                  "--image_path", img, "--output_path", work / f"seg_{tag}_pred.png"], env)
    return (f"segmentation-{tag}", ok and ckpt is not None)


def main():
    env = os.environ.copy()
    env.update({"WANDB_DISABLED": "true", "TQDM_DISABLE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "HF_HOME": os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))})
    work = Path(tempfile.mkdtemp(prefix="dl_smoke_"))
    print(f"Work dir: {work}\n")

    results = []
    for label, fn in [
        ("classification", lambda: smoke_classification(work, env)),
        ("detection-DETR", lambda: smoke_detection_hf(work, env)),
        ("detection-YOLO", lambda: smoke_detection_yolo(work, env)),
        ("segmentation-SegFormer", lambda: smoke_segmentation(work, env, SEG_HF_MODEL, "sf")),
        ("segmentation-UNet", lambda: smoke_segmentation(work, env, SEG_SMP_MODEL, "unet")),
    ]:
        print(f"=== {label} ===", flush=True)
        try:
            results.append(fn())
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            results.append((label, False))
        print()

    if not os.environ.get("SMOKE_KEEP"):
        import shutil
        shutil.rmtree(work, ignore_errors=True)

    print("=" * 40)
    all_ok = True
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        all_ok = all_ok and ok
    print("=" * 40)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
