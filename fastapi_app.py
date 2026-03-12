# fastapi_app.py

import sys
import subprocess
import uuid
import re
import json
import os
import shutil
import psutil
import zipfile
import tempfile
from pydantic import BaseModel, Field
from fastapi import FastAPI, Form, File, UploadFile, Query, Depends, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any, List, Optional

# Import the job_store and Celery tasks
import job_store
from celery_app import run_training_task, process_data_task
import model_registry

# --- Setup ---
# --- 1. Define paths for our SHARED VOLUMES ---
# These paths match the mount points in docker-compose.yml
LOG_DIR_VOL = "/tmp/training_logs" 
DATA_ROOT_VOL = "app_data"
MODEL_OUTPUT_ROOT_VOL = os.path.abspath("model_outputs")

# --- 2. Define configuration variables using the shared volumes ---
LOG_DIR = LOG_DIR_VOL
DATA_ROOT = DATA_ROOT_VOL
MODEL_OUTPUT_ROOT = MODEL_OUTPUT_ROOT_VOL

# UPLOAD_DIR is now inside the shared 'app_data' volume
UPLOAD_DIR = os.path.join(DATA_ROOT, "raw_uploads") 

# OUTPUT_DIR is now inside the shared 'model_outputs' volume
OUTPUT_DIR = os.path.join(MODEL_OUTPUT_ROOT, "inference_outputs") 

PROCESSED_DATA_DIR = os.path.join(DATA_ROOT, "processed_datasets")

# --- 3. Create all directories on startup ---
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_ROOT, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

app = FastAPI(
    title="APHRC No-Code Deep Learning API",
    description="An API for Object Detection, Image Classification and Semantic Segmentation.",
    version="11.0.0",
)

# --- 4. Mount the StaticFiles directory to the *new* shared path ---
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Security ---
API_KEY = os.getenv("API_KEY", "aphrc-secret-key-123") # Default for dev

async def verify_api_key(x_api_key: str = Header(None)):
    """
    Validates the X-API-Key header.
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key

@app.on_event("startup")
def on_startup():
    """Initializes the database on app startup."""
    job_store.init_db()

# --- Helper Functions ---
async def start_job_internal(
    job_id: str,
    task_type: str,
    script_name: str,
    args: List[str],
    output_dir: str,
    metrics_filename: str,
    job_params: Optional[Dict[str, Any]] = None # New Argument
):
    """
    Internal helper to create a job entry in the DB and enqueue the Celery task.
    """
    log_file = os.path.join(LOG_DIR, f"{job_id}.log")
    
    # Merge basic details with full params
    job_details = job_params.copy() if job_params else {}
    
    # Ensure critical fields exist (overwrite if necessary)
    job_details["script_name"] = script_name
    if "run_name" not in job_details:
        job_details["run_name"] = args[args.index("--run_name") + 1] if "--run_name" in args else "N/A"
    if "model_checkpoint" not in job_details:
        job_details["model_checkpoint"] = args[args.index("--model_checkpoint") + 1] if "--model_checkpoint" in args else "N/A"
    if "version" not in job_details:
        job_details["version"] = args[args.index("--version") + 1] if "--version" in args else "N/A"
    
    result = job_store.create_job(
        job_id=job_id,
        task_type=task_type,
        log_file=log_file,
        output_dir=output_dir,
        metrics_filename=metrics_filename,
        details=job_details
    )
    
    cmd = [sys.executable, "-u", script_name] + args
    
    run_training_task.delay(job_id=job_id, cmd=cmd)
    
    return result

# --- Model Registry Endpoint ---
@app.get("/models/list", summary="Get Model Registry")
async def get_model_registry():
    """
    Returns the full dictionary of supported models, grouped by task.
    """
    return model_registry.get_registry()

# --- Data Workspace Endpoints ---
@app.post("/data/upload/{task_type}", summary="Upload a New Dataset", dependencies=[Depends(verify_api_key)])
async def upload_dataset(
    task_type: str,
    data_name: str = Form(..., description="A unique name for the new dataset."),
    data_file: UploadFile = File(..., description="The file containing the dataset (.zip, .csv, .parquet).")
):
    """
    Uploads a new dataset, registers it, and starts the async
    preprocessing task in Celery.
    """
    valid_tasks = [
        "object_detection", 
        "image_classification", 
        "semantic_segmentation"
    ]
    
    if task_type not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"Invalid task_type. Must be one of {valid_tasks}")
        
    # Get extension (e.g. .zip, .csv)
    filename = data_file.filename
    ext = os.path.splitext(filename)[1].lower()
    
    temp_file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{filename}")
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(data_file.file, buffer)

    dataset_id = str(uuid.uuid4())
    processed_path = os.path.join(PROCESSED_DATA_DIR, dataset_id)
    
    try:
        dataset = job_store.create_dataset_entry(
            name=data_name,
            task_type=task_type,
            disk_path=processed_path
        )
    except Exception as e:
        if "UNIQUE constraint failed" in str(e):
            raise HTTPException(status_code=400, detail=f"A dataset named '{data_name}' for task '{task_type}' already exists.")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    process_data_task.delay(
        dataset_id=dataset.id,
        file_path=temp_file_path,  # Renamed from zip_path
        task_type=task_type
    )
    
    return {"dataset_id": dataset.id, "name": dataset.name, "status": "processing"}

@app.get("/data/list/{task_type}", response_model=List[Dict[str, str]], summary="List Ready Datasets")
async def get_dataset_list(task_type: str):
    """
    Gets a list of all 'ready' datasets for a given task type.
    Used to populate the 'Select Dataset' dropdowns in the UI.
    """
    return job_store.list_datasets_by_task(task_type)

@app.get("/data/status/{dataset_id}", summary="Get Dataset Processing Status")
async def get_dataset_status(dataset_id: str):
    """
    Polls the processing status of a single dataset upload.
    """
    dataset = job_store.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {
        "id": dataset.id,
        "name": dataset.name,
        "status": dataset.status,
        "error": dataset.error_message
    }

# --- Training Endpoints ---

@app.post("/train/object-detection", status_code=202, summary="Start Object Detection Job", dependencies=[Depends(verify_api_key)])
async def start_object_detection_training(
    # --- Common parameters ---
    dataset_id: str = Form(..., description="ID of the registered dataset to use"),
    model_checkpoint: str = Form(..., description="Full model name from the registry"), 
    run_name: str = Form("shiny-obj-run", description="Name for the training run"),
    version: str = Form("1.0.0", description="Version string for the run"),
    epochs: int = Form(5),
    train_batch_size: int = Form(8),
    seed: int = Form(42),
    early_stopping_patience: int = Form(5),
    max_image_size: int = Form(640),
    
    # HF Params
    eval_batch_size: int = Form(8),
    learning_rate: float = Form(5e-5),
    weight_decay_hf: float = Form(1e-4),
    gradient_accumulation_steps: int = Form(1),
    gradient_checkpointing: bool = Form(False),
    early_stopping_threshold: float = Form(0.0),
    
    # HF Params (Added for Production Audit)
    optimizer_hf: str = Form("adamw_torch"),
    scheduler_hf: str = Form("linear"), 
    warmup_epochs_hf: float = Form(1.0),
    
    # YOLO Params
    warmup_epochs: float = Form(3.0),
    lr0: float = Form(0.01),
    momentum: float = Form(0.937),
    optimizer: str = Form("auto"),
    weight_decay_yolo: float = Form(0.0005),
    
    # Augmentation
    enable_augmentation: bool = Form(False),
    flip_prob: float = Form(0.5),
    rotate_limit: int = Form(15),
    brightness: float = Form(0.2),
    contrast: float = Form(0.2),
    mosaic: float = Form(1.0),
    mixup: float = Form(0.0),
    hsv_h: float = Form(0.015),
    hsv_s: float = Form(0.7),
    hsv_v: float = Form(0.4)
):
    """
    Start an Object Detection Training Job (YOLO or Hugging Face).
    """
    task_type = "object_detection"
    
    # --- 1. Validation ---
    allowed_models = model_registry.get_task_models(task_type)
    all_checkpoints = [chk for arch_models in allowed_models.values() for chk in arch_models]
    # Allow manual override or check registry
    if model_checkpoint not in all_checkpoints:
        raise HTTPException(status_code=400, detail=f"Invalid model_checkpoint. Not in registry.")

    job_id = str(uuid.uuid4())
    
    dataset = job_store.get_dataset(dataset_id)
    if not dataset or dataset.status != "ready" or dataset.task_type != task_type:
        raise HTTPException(status_code=400, detail="Dataset not ready or valid.")
    
    processed_data_path = dataset.disk_path 
    
    # --- 2. Routing Logic ---
    
    # Include version in output_dir to match script logic and prevent mismatch
    output_dir_name = f"{model_checkpoint.split('/')[-1]}-{run_name}-{version}"
    output_dir = os.path.join(MODEL_OUTPUT_ROOT, output_dir_name)
    metrics_filename = f"metrics_{job_id}.json"
    
    # --- System Defaults ---
    # fp16 = torch.cuda.is_available()
    num_proc = 4
    
    # --- Base arguments common to both scripts ---
    # Define processed data location (Arrow format for HF)
    arrow_cache_dir = os.path.join(processed_data_path, "processed_hf")
    
    base_args = [
        "--data_dir", processed_data_path,
        "--processed_data_dir", arrow_cache_dir,
        "--model_checkpoint", model_checkpoint,
        "--run_name", run_name,
        "--version", version,
        "--epochs", str(epochs),
        "--train_batch_size", str(train_batch_size),
        "--seed", str(seed),
        "--early_stopping_patience", str(early_stopping_patience),
        "--metrics_filename", metrics_filename,
        "--model_output_root", MODEL_OUTPUT_ROOT, 
        "--max_image_size", str(max_image_size),
        "--num_proc", str(num_proc), 
    ]

    # Augmentation Args
    if enable_augmentation:
        base_args.append("--enable_augmentation")
        base_args.extend(["--flip_prob", str(flip_prob)])
        base_args.extend(["--rotate_limit", str(rotate_limit)])
        base_args.extend(["--brightness", str(brightness)])
        base_args.extend(["--contrast", str(contrast)])

    # Routing Logic: Ultralytics (YOLO/RT-DETR) vs Transformers (YOLOS/DETR)
    chk_lower = model_checkpoint.lower()
    # YOLOS is a Transformer model, so we must explicitly exclude it from the YOLO check
    is_ultralytics_model = (("yolo" in chk_lower and "yolos" not in chk_lower) or "rtdetr" in chk_lower)
    
    if is_ultralytics_model:
# --- Build YOLO/Ultralytics Argument List ---
        print(f"Detected Ultralytics model (YOLO/RT-DETR). Routing to object_detection_train_yolo.py")
        script_name = "object_detection_train_yolo.py"
        
        yolo_args = [
            "--output_dir", output_dir, # Explicitly passed here for YOLO
            "--warmup_epochs", str(warmup_epochs),
            "--lr0", str(lr0),
            "--momentum", str(momentum),
            "--optimizer", optimizer,
            "--weight_decay", str(weight_decay_yolo),
            # YOLO Augmentations
            "--mosaic", str(mosaic),
            "--mixup", str(mixup),
            "--hsv_h", str(hsv_h),
            "--hsv_s", str(hsv_s),
            "--hsv_v", str(hsv_v)
        ]
        args = base_args + yolo_args
        
    else:
        # --- Build Hugging Face Transformers Argument List ---
        print(f"Detected Transformers model. Routing to object_detection_train.py")
        script_name = "object_detection_train.py"
        
        hf_args = [
            "--learning_rate", str(learning_rate),
            "--weight_decay", str(weight_decay_hf),
            "--gradient_accumulation_steps", str(gradient_accumulation_steps),
            "--early_stopping_threshold", str(early_stopping_threshold),
            "--early_stopping_threshold", str(early_stopping_threshold),
            "--eval_batch_size", str(eval_batch_size),
            "--optimizer", optimizer_hf,
            "--scheduler", scheduler_hf,
            "--warmup_epochs", str(warmup_epochs_hf),
        ]
        args = base_args + hf_args
        
        # Add HF-specific booleans
        # if fp16: args.append("--fp16") # Managed in scripts
        if gradient_checkpointing: args.append("--gradient_checkpointing")
        
    # --- 3. Construct Context-Aware Job Parameters ---
    # Only store parameters relevant to the selected model architecture
    
    # Common Parameters
    common_params = {
        "dataset_id": dataset_id, "model_checkpoint": model_checkpoint, "run_name": run_name,
        "version": version, "epochs": epochs, "train_batch_size": train_batch_size, 
        "seed": seed, "early_stopping_patience": early_stopping_patience, "max_image_size": max_image_size,
        "is_yolo_model": is_ultralytics_model, "num_proc": num_proc, "enable_augmentation": enable_augmentation,
        # Common Augmentation
        "flip_prob": flip_prob, "rotate_limit": rotate_limit, "brightness": brightness, "contrast": contrast
    }

    if is_ultralytics_model:
        # YOLO Specific Params
        specific_params = {
            "lr0": lr0, 
            "momentum": momentum, 
            "optimizer": optimizer, 
            "weight_decay": weight_decay_yolo,
            "warmup_epochs": warmup_epochs,
            # YOLO Augmentations
            "mosaic": mosaic, "mixup": mixup, "hsv_h": hsv_h, "hsv_s": hsv_s, "hsv_v": hsv_v
        }
    else:
        # Transformers Specific Params
        specific_params = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay_hf,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_checkpointing": gradient_checkpointing,
            "early_stopping_threshold": early_stopping_threshold,
            "eval_batch_size": eval_batch_size,
            "optimizer": optimizer_hf,
            "scheduler": scheduler_hf,
            "warmup_epochs": warmup_epochs_hf
        }
        
    job_params = {**common_params, **specific_params}

    return await start_job_internal(job_id, task_type, script_name, args, output_dir, metrics_filename, job_params=job_params)




@app.post("/train/image-classification", status_code=202, summary="Start Image Classification Job", dependencies=[Depends(verify_api_key)])
async def start_image_classification_training(
    dataset_id: str = Form(...),
    model_checkpoint: str = Form(...), 
    run_name: str = Form("shiny-img-class-run"),
    version: str = Form("1.0.0"),
    is_presplit: bool = Form(True),
    train_ratio: float = Form(0.8),
    dev_ratio: float = Form(0.1),
    epochs: int = Form(5),
    learning_rate: float = Form(2e-5),
    weight_decay: float = Form(0.01),
    train_batch_size: int = Form(32),
    eval_batch_size: int = Form(32),
    max_image_size: int = Form(224),
    gradient_accumulation_steps: int = Form(1),
    gradient_checkpointing: bool = Form(False),
    seed: int = Form(42),
    early_stopping_patience: int = Form(3),
    early_stopping_threshold: float = Form(0.0),
    
    # Added for Production Audit
    optimizer: str = Form("adamw_torch"),
    scheduler: str = Form("linear"),
    warmup_epochs: float = Form(1.0),
    
    # Augmentation
    enable_augmentation: bool = Form(False),
    flip_prob: float = Form(0.5),
    rotate_limit: int = Form(15),
    brightness: float = Form(0.2),
    contrast: float = Form(0.2)
):
    """
    Starts a new Image Classification training job.
    """
    task_type = "image_classification"
    allowed_models = model_registry.get_task_models(task_type)
    all_checkpoints = [chk for arch_models in allowed_models.values() for chk in arch_models]
    if model_checkpoint not in all_checkpoints:
        raise HTTPException(status_code=400, detail=f"Invalid model_checkpoint. Not in registry.")

    job_id = str(uuid.uuid4())

    dataset = job_store.get_dataset(dataset_id)
    if not dataset or dataset.status != "ready" or dataset.task_type != task_type:
        raise HTTPException(status_code=400, detail="Dataset not ready or not valid for this task")

    unzip_dir = dataset.disk_path

    output_dir_name = f"{model_checkpoint.split('/')[-1]}-{run_name}-{version}"
    output_dir = os.path.join(MODEL_OUTPUT_ROOT, output_dir_name)
    metrics_filename = f"metrics_{job_id}.json"
    
    # --- System Defaults ---
    # fp16 = torch.cuda.is_available()
    num_proc = 4
    
    args = [
        "--data_dir", unzip_dir,
        "--model_checkpoint", model_checkpoint,
        "--run_name", run_name,
        "--version", version,
        "--epochs", str(epochs),
        "--learning_rate", str(learning_rate),
        "--weight_decay", str(weight_decay),
        "--train_batch_size", str(train_batch_size),
        "--eval_batch_size", str(eval_batch_size),
        "--max_image_size", str(max_image_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--seed", str(seed),
        "--early_stopping_patience", str(early_stopping_patience),
        "--early_stopping_threshold", str(early_stopping_threshold),
        "--metrics_filename", metrics_filename,
        "--model_output_root", MODEL_OUTPUT_ROOT,
        "--num_proc", str(num_proc),
        "--optimizer", optimizer,
        "--scheduler", scheduler,
        "--warmup_epochs", str(warmup_epochs),
    ]
    if gradient_checkpointing: args.append("--gradient_checkpointing")
    # if fp16: args.append("--fp16") # Managed in scripts
    if is_presplit: args.append("--is_presplit")
    else:
        args.extend(["--train_ratio", str(train_ratio)])
        args.extend(["--dev_ratio", str(dev_ratio)])
        
    # Augmentation
    if enable_augmentation:
        args.append("--enable_augmentation")
        args.extend(["--flip_prob", str(flip_prob)])
        args.extend(["--rotate_limit", str(rotate_limit)])
        args.extend(["--brightness", str(brightness)])
        args.extend(["--contrast", str(contrast)])

    job_params = {
        "dataset_id": dataset_id, "model_checkpoint": model_checkpoint, "run_name": run_name,
        "version": version, "is_presplit": is_presplit, "train_ratio": train_ratio,
        "dev_ratio": dev_ratio, "epochs": epochs, "learning_rate": learning_rate,
        "weight_decay": weight_decay, "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size, "max_image_size": max_image_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_checkpointing": gradient_checkpointing, "seed": seed,
        "early_stopping_patience": early_stopping_patience, "early_stopping_threshold": early_stopping_threshold, "num_proc": num_proc,
        "optimizer": optimizer, "scheduler": scheduler, "warmup_epochs": warmup_epochs,
        "enable_augmentation": enable_augmentation, "flip_prob": flip_prob,
        "rotate_limit": rotate_limit, "brightness": brightness, "contrast": contrast
    }

    return await start_job_internal(job_id, task_type, "image_classification_train.py", args, output_dir, metrics_filename, job_params=job_params)

@app.post("/train/image-segmentation", status_code=202, summary="Start Image Segmentation Job", dependencies=[Depends(verify_api_key)])
async def start_semantic_segmentation_training(
    dataset_id: str = Form(...),
    model_checkpoint: str = Form(...), 
    run_name: str = Form("shiny-seg-run"),
    version: str = Form("1.0.0"),
    is_presplit: bool = Form(True),
    train_ratio: float = Form(0.8),
    dev_ratio: float = Form(0.1),
    epochs: int = Form(10),
    learning_rate: float = Form(6e-5),
    weight_decay: float = Form(0.01),
    train_batch_size: int = Form(8),
    eval_batch_size: int = Form(8),
    max_image_size: int = Form(512),
    gradient_accumulation_steps: int = Form(1),
    gradient_checkpointing: bool = Form(False),
    seed: int = Form(42),
    early_stopping_patience: int = Form(5),
    early_stopping_threshold: float = Form(0.0),
    
    # Added for Production Audit
    optimizer: str = Form("adamw_torch"),
    scheduler: str = Form("linear"),
    warmup_epochs: float = Form(1.0),
    
    # Augmentation (Added Phase 6)
    enable_augmentation: bool = Form(False),
    flip_prob: float = Form(0.5),
    rotate_limit: int = Form(15),
    brightness: float = Form(0.2),
    contrast: float = Form(0.2)
):
    """
    Start a Semantic Segmentation Training Job.
    """
    task_type = "semantic_segmentation"
    allowed_models = model_registry.get_task_models(task_type)
    all_checkpoints = [chk for arch_models in allowed_models.values() for chk in arch_models]
    if model_checkpoint not in all_checkpoints:
        raise HTTPException(status_code=400, detail=f"Invalid model_checkpoint. Not in registry.")

    job_id = str(uuid.uuid4())
    dataset = job_store.get_dataset(dataset_id)
    if not dataset or dataset.status != "ready" or dataset.task_type != task_type:
        raise HTTPException(status_code=400, detail="Dataset not ready or valid.")
    
    output_dir_name = f"{model_checkpoint.split('/')[-1]}-{run_name}-{version}"
    output_dir = os.path.join(MODEL_OUTPUT_ROOT, output_dir_name)
    metrics_filename = f"metrics_{job_id}.json"
    
    # --- System Defaults ---
    # fp16 = torch.cuda.is_available()
    num_proc = 4
    
    args = [
        "--data_dir", dataset.disk_path,
        "--model_checkpoint", model_checkpoint,
        "--output_dir", output_dir,
        "--epochs", str(epochs),
        "--learning_rate", str(learning_rate),
        "--weight_decay", str(weight_decay),
        "--train_batch_size", str(train_batch_size),
        "--eval_batch_size", str(eval_batch_size),
        "--max_image_size", str(max_image_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--seed", str(seed),
        "--early_stopping_patience", str(early_stopping_patience),
        "--early_stopping_threshold", str(early_stopping_threshold),
        "--metrics_filename", metrics_filename,
        "--model_output_root", MODEL_OUTPUT_ROOT,
        "--num_proc", str(num_proc),
        "--optimizer", optimizer,
        "--scheduler", scheduler,
        "--warmup_epochs", str(warmup_epochs),
    ]
    if gradient_checkpointing: args.append("--gradient_checkpointing")
    # if fp16: args.append("--fp16") # Managed in script
    if is_presplit: args.append("--is_presplit")
    else:
        args.extend(["--train_ratio", str(train_ratio)])
        args.extend(["--dev_ratio", str(dev_ratio)])
        
    # Augmentation
    if enable_augmentation:
        args.append("--enable_augmentation")
        args.extend(["--flip_prob", str(flip_prob)])
        args.extend(["--rotate_limit", str(rotate_limit)])
        args.extend(["--brightness", str(brightness)])
        args.extend(["--contrast", str(contrast)])

    job_params = {
        "dataset_id": dataset_id, "model_checkpoint": model_checkpoint, "run_name": run_name,
        "version": version, "is_presplit": is_presplit, "train_ratio": train_ratio,
        "dev_ratio": dev_ratio, "epochs": epochs, "learning_rate": learning_rate,
        "weight_decay": weight_decay, "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size, "max_image_size": max_image_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_checkpointing": gradient_checkpointing, "seed": seed,
        "early_stopping_patience": early_stopping_patience, "early_stopping_threshold": early_stopping_threshold, "num_proc": num_proc,
        "optimizer": optimizer, "scheduler": scheduler, "warmup_epochs": warmup_epochs,
        "enable_augmentation": enable_augmentation, "flip_prob": flip_prob,
        "rotate_limit": rotate_limit, "brightness": brightness, "contrast": contrast
    }

    return await start_job_internal(job_id, task_type, "image_segmentation_train.py", args, output_dir, metrics_filename, job_params=job_params)



# --- Status & Metrics Endpoints ---
@app.get("/status/{job_id}", summary="Get Job Status, Logs, and Progress")
async def get_job_status(job_id: str):
    """
    Polls for the status of an active training job.
    Returns the job status, full log contents, and progress percentage.
    """
    job = job_store.get_job(job_id)
    if not job: 
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    
    log_content = job_store.read_log_file(job_id, tail_lines=100)
    
    progress = {"percentage": 0, "text": "Initializing..."}
    
    # Progress Calculation from active metrics payload
    metrics_path = os.path.join(job.output_dir, job.metrics_filename) if job.output_dir and job.metrics_filename else None
    
    if metrics_path and os.path.exists(metrics_path):
        try:
            # Read last line of metrics
            with open(metrics_path, "rb") as f:
                try:
                    f.seek(-1024, os.SEEK_END) # Go to end
                except OSError:
                    f.seek(0) # File too small
                    
                last_lines = f.readlines()
                if last_lines:
                    last_metric = json.loads(last_lines[-1].decode("utf-8", errors="ignore"))
                    
                    # Calculate progress
                    current_epoch = last_metric.get("epoch", 0)
                    total_epochs = int(job.job_details.get("epochs", 1))
                    if total_epochs > 0:
                        pct = min(int((current_epoch / total_epochs) * 100), 100)
                        progress["percentage"] = pct
                        progress["text"] = f"Epoch {current_epoch:.2f}/{total_epochs}"
        except Exception:
             # Fallback to log parsing if metrics fail
             pass

    # Fallback: Log Parsing (if metrics didn't yield progress)
    if progress["percentage"] == 0:
        lines = log_content.splitlines()
        # ANSI escape code regex
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        progress_regex = re.compile(r"(\d+)\s*%")
        
        for line in reversed(lines):
            # Clean line
            clean_line = ansi_escape.sub('', line).strip()
            if not clean_line: continue
            
            # Skip download/extraction progress bars to avoid "Starts Full" issue
            if "Downloading" in clean_line or "Extracting" in clean_line:
                continue
                
            progress_match = progress_regex.search(clean_line)
            if progress_match:
                try:
                    pct = int(progress_match.group(1))
                    # Only update if valid pct
                    if 0 <= pct <= 100:
                        progress["percentage"] = pct
                        progress["text"] = clean_line
                        break
                except ValueError:
                    continue
            
    return {"status": job.status, "task": job.task_type, "log": log_content, "progress": progress}

@app.get("/metrics/{job_id}", response_model=List[Dict[str, Any]], summary="Get Job Metrics")
async def get_job_metrics(job_id: str):
    """
    Fetches the evaluation metrics for a given job.
    Returns a list of JSON objects, one for each evaluation step.
    """
    job = job_store.get_job(job_id)
    if not job or not job.output_dir or not job.metrics_filename: 
        return []
        
    metrics_file_path = os.path.join(job.output_dir, job.metrics_filename)
    
    if not os.path.exists(metrics_file_path): 
        return []
        
    metrics_data = []
    with open(metrics_file_path, "r") as f:
        for line in f:
            try:
                metrics_data.append(json.loads(line.strip()))
            except json.JSONDecodeError: 
                continue
    return metrics_data

# --- Hardened Inference & Checkpoint Endpoints ---
@app.get("/checkpoints", response_model=List[str], summary="Find Checkpoints for Inference")
async def find_checkpoints(
    run_name: str = Query(..., min_length=1, description="The 'run_name' of the training job."),
    task_type: str = Query(..., description="The task type (e.g., 'object_detection')") 
):
    """
    Safely finds all completed checkpoint folders for a given run_name and task_type.
    """
    if task_type not in ["object_detection", "image_classification", "semantic_segmentation"]:
        raise HTTPException(status_code=400, detail="Invalid task_type")
        
    found_checkpoints = job_store.get_valid_checkpoints(
        task_type=task_type, 
        run_name=run_name
    )
    return found_checkpoints

@app.get("/jobs/list", response_model=List[Dict[str, Any]], summary="List All Jobs")
async def get_job_list(
    task_type: Optional[str] = Query(None, description="Filter by task type."),
    status: Optional[str] = Query(None, description="Filter by job status (completed, failed, etc.)")
):
    """
    Gets a list of all jobs from the database, with optional filters,
    for the 'Training History' tab.
    """
    if status and status not in ["completed", "failed", "running", "queued"]:
        status = None
    return job_store.list_all_jobs(task_type=task_type, status=status)

@app.post("/jobs/cancel/{job_id}", summary="Cancel a Running Job", dependencies=[Depends(verify_api_key)])
async def cancel_job(job_id: str):
    """
    Cancels a running job by terminating its subprocess.
    """
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    if job.status not in ["running", "queued"]:
        return {"message": "Job is not running", "status": job.status}

    pid = job.pid
    if not pid:
        # If no PID (e.g. queued or legacy job), just mark as cancelled
        job_store.update_job_status(job_id, "cancelled")
        return {"message": "Job cancelled (no PID found)", "status": "cancelled"}

    try:
        import signal
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to process {pid} for job {job_id}")
        
        # Verify if it's dead? For now, we assume it works and update status.
        job_store.update_job_status(job_id, "cancelled")
        
        return {"message": f"Job {job_id} cancelled (PID {pid} terminated)", "status": "cancelled"}
        
    except ProcessLookupError:
        # Process already gone
        job_store.update_job_status(job_id, "cancelled") # or failed/completed?
        return {"message": "Job process not found, marked as cancelled", "status": "cancelled"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {e}")


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {e}")

@app.delete("/jobs/{job_id}", summary="Delete a Job and its Outputs", dependencies=[Depends(verify_api_key)])
async def delete_job_endpoint(job_id: str):
    """
    Deletes a job from the database AND its associated output files/logs.
    Only allowed if the job is NOT running.
    """
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    if job.status in ["running"]:
        raise HTTPException(status_code=400, detail="Cannot delete a running job. Cancel it first.")

    paths = job_store.delete_job(job_id)
    if not paths:
        raise HTTPException(status_code=500, detail="Failed to delete job from database")

    # Clean up files
    deleted_files = []
    errors = []
    
    # Delete Log File
    log_file = paths.get("log_file")
    if log_file and os.path.exists(log_file):
        try:
            os.remove(log_file)
            deleted_files.append("log_file")
        except Exception as e:
            errors.append(f"Failed to delete log file: {e}")

    # Delete Output Directory
    output_dir = paths.get("output_dir")
    if output_dir and os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
            deleted_files.append("output_dir")
        except Exception as e:
            errors.append(f"Failed to delete output dir: {e}")

    return {
        "message": f"Job {job_id} deleted.",
        "deleted_components": deleted_files,
        "cleanup_errors": errors
    }

@app.delete("/data/{dataset_id}", summary="Delete a Registered Dataset", dependencies=[Depends(verify_api_key)])
async def delete_dataset_endpoint(dataset_id: str):
    """
    Deletes a dataset from the database AND its files on disk.
    """
    dataset = job_store.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    disk_path = job_store.delete_dataset(dataset_id)
    if not disk_path: 
        raise HTTPException(status_code=500, detail="Failed to delete dataset from database")

    # Clean up disk files
    cleanup_status = "Skipped (not found)"
    if disk_path and os.path.exists(disk_path):
        try:
            shutil.rmtree(disk_path)
            cleanup_status = "Success"
        except Exception as e:
            cleanup_status = f"Failed: {e}"
    return {
        "message": f"Dataset {dataset_id} deleted.",
        "disk_path": disk_path,
        "cleanup_status": cleanup_status
    }

@app.get("/system/health", summary="Get System Health Metrics")
async def get_system_health():
    """
    Returns CPU, RAM, Disk, and GPU usage.
    """
    # CPU
    cpu_percent = psutil.cpu_percent(interval=None)
    
    # RAM
    mem = psutil.virtual_memory()
    ram_percent = mem.percent
    ram_used_gb = round(mem.used / (1024**3), 2)
    ram_total_gb = round(mem.total / (1024**3), 2)
    
    # Disk (where models are stored) -- check if path exists to avoid error
    disk_path = MODEL_OUTPUT_ROOT if os.path.exists(MODEL_OUTPUT_ROOT) else "/" 
    disk = psutil.disk_usage(disk_path)
    disk_percent = disk.percent
    disk_free_gb = round(disk.free / (1024**3), 2)
    
    # GPU
    gpu_stats = []
    try:
        # Simple nvidia-smi parsing
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        for line in result.strip().split('\n'):
            parts = line.split(',')
            if len(parts) == 5:
                idx, name, util, mem_used, mem_total = [x.strip() for x in parts]
                gpu_stats.append({
                    "index": int(idx),
                    "name": name,
                    "utilization": float(util),
                    "memory_used_mb": float(mem_used),
                    "memory_total_mb": float(mem_total)
                })
    except Exception:
        # GPU not available or nvidia-smi failed
        # Just return empty list
        pass
        
    return {
        "cpu": {"percent": cpu_percent},
        "ram": {"percent": ram_percent, "used_gb": ram_used_gb, "total_gb": ram_total_gb},
        "disk": {"percent": disk_percent, "free_gb": disk_free_gb},
        "gpu": gpu_stats
    }

@app.post("/inference/object-detection", summary="Run Object Detection Inference")
async def run_object_detection_inference(
    model_checkpoint: str = Form(..., description="Path to the model checkpoint, (e.g., 'model_outputs/...')"),
    image: UploadFile = File(..., description="The image to run inference on."),
    threshold: float = Form(0.25, description="The confidence threshold for predictions (used as 'conf' for YOLO)."),
    iou: float = Form(0.7, description="The IoU threshold for NMS (YOLO-only)."),
    max_det: int = Form(300, description="Maximum number of detections (YOLO-only)."),
    imgsz: int = Form(640, description="Inference image size (YOLO-only)."),
    classes: str = Form(None, description="Filter by class (e.g., '0,2'). Empty for all. (YOLO-only).")
):
    """
    --- ROUTER (V2) ---
    Runs inference for object detection.
    Routes to YOLO, YOLOS, or Transformers script based on the model checkpoint.
    """
    if not job_store.is_valid_checkpoint_path(model_checkpoint):
        raise HTTPException(status_code=400, detail="Invalid or malicious model path")
    
    try:
        input_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{image.filename}")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        output_filename = f"output_{os.path.basename(input_path)}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # --- NEW ROUTING LOGIC (3-WAY) ---
        yolo_weights_path = os.path.join(model_checkpoint, "weights", "best.pt")

        if os.path.exists(yolo_weights_path) or model_checkpoint.endswith(".pt"):
            # 1. Handle .pt YOLO models
            print("Routing to YOLO inference: object_detection_inference_yolo.py")
            script_name = "object_detection_inference_yolo.py"
            cmd_args = [
                "--threshold", str(threshold),
                "--iou", str(iou),
                "--max_det", str(max_det),
                "--imgsz", str(imgsz)
            ]
            if classes:
                cmd_args.extend(["--classes", classes])
                
        elif "yolos" in model_checkpoint.lower():
            # 2. Handle YOLOS (transformers) models
            print("Routing to YOLOS inference: object_detection_inference_yolos.py")
            script_name = "object_detection_inference_yolos.py"
            cmd_args = [
                # YOLOS script uses a higher default (0.9), but we
                # pass the user's value for consistency.
                "--threshold", str(threshold)
            ]
        else:
            # 3. Handle default (DETR, etc.) transformers models
            print("Routing to Transformers (DETR/default) inference: object_detection_inference.py")
            script_name = "object_detection_inference.py"
            cmd_args = [
                "--threshold", str(threshold)
            ]
        # --- END ROUTING LOGIC ---

        cmd = [
            sys.executable, script_name,
            "--model_checkpoint", model_checkpoint,
            "--image_path", input_path,
            "--output_path", output_path
        ] + cmd_args
        
        process = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {"output_url": f"/outputs/{output_filename}"}

    except subprocess.CalledProcessError as e:
        return JSONResponse(status_code=500, content={"error": "Inference script failed", "details": e.stderr})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/inference/asr", summary="Run ASR Inference")
async def run_asr_inference_endpoint(
    model_checkpoint: str = Form(..., description="Path to the model checkpoint"),
    audio: UploadFile = File(..., description="The audio file to transcribe.")
):
    """
    Runs the 'asr_inference.py' script on an uploaded audio file
    and returns the transcription as a string.
    """
    if not job_store.is_valid_checkpoint_path(model_checkpoint):
        raise HTTPException(status_code=400, detail="Invalid or malicious model path")
        
    try:
        input_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{audio.filename}")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        cmd = [
            sys.executable, "asr_inference.py",
            "--model_checkpoint", model_checkpoint,
            "--audio_path", input_path,
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True, check=True)
        transcription = process.stdout.strip()
        
        return {"transcription": transcription}

    except subprocess.CalledProcessError as e:
        return JSONResponse(status_code=500, content={"error": "ASR inference script failed", "details": e.stderr})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.post("/inference/image-classification", summary="Run Image Classification Inference")
async def run_image_classification_inference(
    model_checkpoint: str = Form(..., description="Path to the model checkpoint"),
    image: UploadFile = File(..., description="The image to classify.")
):
    """
    Runs the 'image_classification_inference.py' script on an uploaded image
    and returns the predicted class as a string.
    """
    if not job_store.is_valid_checkpoint_path(model_checkpoint):
        raise HTTPException(status_code=400, detail="Invalid or malicious model path")
        
    try:
        input_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{image.filename}")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        cmd = [
            sys.executable, "image_classification_inference.py",
            "--model_checkpoint", model_checkpoint,
            "--image_path", input_path,
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True, check=True)
        prediction = process.stdout.strip()
        
        return {"prediction": prediction}
    except subprocess.CalledProcessError as e:
        return JSONResponse(status_code=500, content={"error": "Inference script failed", "details": e.stderr})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/inference/image-segmentation", summary="Run Image Segmentation Inference")
async def run_semantic_segmentation_inference(
    model_checkpoint: str = Form(..., description="Path to the model checkpoint"),
    image: UploadFile = File(..., description="The image to segment.")
):
    """
    Runs the 'semantic_segmentation_inference.py' script on an uploaded image
    and returns a URL to the annotated output image.
    """
    if not job_store.is_valid_checkpoint_path(model_checkpoint):
        raise HTTPException(status_code=400, detail="Invalid or malicious model path")
    
    try:
        input_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{image.filename}")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        output_filename = f"output_{os.path.basename(input_path)}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        cmd = [
            sys.executable, "semantic_segmentation_inference.py",
            "--model_checkpoint", model_checkpoint,
            "--image_path", input_path,
            "--output_path", output_path
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {"output_url": f"/outputs/{output_filename}"}

    except subprocess.CalledProcessError as e:
        return JSONResponse(status_code=500, content={"error": "Inference script failed", "details": e.stderr})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})



# --- Job Log Endpoint ---
@app.get("/jobs/{job_id}/log", summary="Get Job Logs")
async def get_job_log(job_id: str):
    """
    Reads the log file for a specific job and returns it as plain text.
    """
    try:
        job = job_store.get_job(job_id)
        if not job or not job.log_file:
             raise HTTPException(status_code=404, detail="Job or log file not found")
             
        if not os.path.exists(job.log_file):
             return JSONResponse(content={"log": "Log file not created yet."}, status_code=200)
             
        with open(job.log_file, "r") as f:
            content = f.read()
            return JSONResponse(content={"log": content})
            
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- Job Download Endpoint ---
@app.get("/jobs/{job_id}/download", summary="Download Job Outputs")
async def download_job_outputs(job_id: str, background_tasks: BackgroundTasks):
    """
    Zips the output directory of a completed job and returns it as a downloadable file.
    """
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed. Status: {job.status}")
        
    output_dir = job.output_dir
    if not output_dir or not os.path.exists(output_dir):
        raise HTTPException(status_code=404, detail="Job output directory not found")
        
    # Security check: Ensure output_dir is within MODEL_OUTPUT_ROOT
    root_path = os.path.abspath(MODEL_OUTPUT_ROOT)
    full_path = os.path.abspath(output_dir)
    if not full_path.startswith(root_path):
         raise HTTPException(status_code=403, detail="Access denied to this path")

    # Create zip file
    temp_dir = tempfile.gettempdir()
    archive_name = f"model_output_{job_id}"
    archive_path_base = os.path.join(temp_dir, archive_name)
    
    try:
        # shutil.make_archive adds the extension automatically (e.g. .zip)
        zip_path = shutil.make_archive(archive_path_base, 'zip', output_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create zip archive: {e}")

    # Add background task to remove the file after sending
    background_tasks.add_task(os.remove, zip_path)
    
    filename = f"{job.job_details.get('run_name', 'model')}-{job.task_type}.zip"
    
    return FileResponse(
        zip_path, 
        media_type="application/zip", 
        filename=filename
    )