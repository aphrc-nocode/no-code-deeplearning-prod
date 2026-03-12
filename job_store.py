# job_store.py

import os
import json
from collections import deque
import glob
import uuid
from sqlalchemy import create_engine, Column, String, Text, JSON, UniqueConstraint, desc, Integer
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from typing import Dict, Any, Optional, List


# --- Database Setup ---
DATA_ROOT = "app_data" 
os.makedirs(DATA_ROOT, exist_ok=True)
DB_FILE = os.path.join(DATA_ROOT, "jobs.db")
DB_URL = f"sqlite:///{DB_FILE}"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

MODEL_OUTPUT_ROOT = "model_outputs" 

class Base(DeclarativeBase):
    pass

class Job(Base):
    """
    Database model for a single training job.
    """
    __tablename__ = "jobs"
    id = Column(String, primary_key=True, index=True)
    status = Column(String, default="queued")
    task_type = Column(String) # 'asr', 'object_detection', etc.
    log_file = Column(String)
    output_dir = Column(String)
    metrics_filename = Column(String)
    pid = Column(Integer, nullable=True) # Check process ID for cancellation
    job_details = Column(JSON, default={}) # Store run_name, model_checkpoint

class RegisteredDataset(Base):
    """
    Database model for a single registered dataset.
    """
    __tablename__ = "datasets"
    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, index=True, nullable=False)
    task_type = Column(String, nullable=False) 
    status = Column(String, default="processing") # processing, ready, failed
    disk_path = Column(String, unique=True, nullable=False)
    error_message = Column(Text, nullable=True)
    
    __table_args__ = (UniqueConstraint('name', 'task_type', name='_name_task_uc'),)

def init_db():
    """Initialize the database and create tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    
    # --- Database Migrations ---
    # Since we added 'pid' later, we need to ensure it exists in old DBs.
    # We use raw SQL for this simple migration instead of setting up Alembic.
    with engine.connect() as conn:
        try:
            # Try to select the column to see if it exists
            conn.execute("SELECT pid FROM jobs LIMIT 1")
        except Exception:
            # If it fails, add the column
            print("Migrating DB: Adding 'pid' column to 'jobs' table.")
            try:
                conn.execute("ALTER TABLE jobs ADD COLUMN pid INTEGER")
            except Exception as e:
                print(f"Migration warning: {e}")

# --- Job Functions ---

def get_job(job_id: str) -> Optional[Job]:
    """Retrieves a single job object from the database."""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        return job
    finally:
        db.close()


def list_all_jobs(task_type: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """Lists all jobs in the database, with optional filters."""
    db = SessionLocal()
    try:
        query = db.query(Job)
        
        # Apply task filter
        if task_type:
            query = query.filter(Job.task_type == task_type)
        
        # Apply status filter
        if status:
            query = query.filter(Job.status == status)
            
        jobs = query.order_by(desc(Job.id)).all()
        
        return [
            {
                "id": job.id,
                "task_type": job.task_type,
                "status": job.status,
                "details": job.job_details,
                "version": job.job_details.get("version", "N/A"),
                "run_name": job.job_details.get("run_name", "N/A")
            }
            for job in jobs
        ]
    finally:
        db.close()


def create_job(
    job_id: str, 
    task_type: str, 
    log_file: str, 
    output_dir: str, 
    metrics_filename: str, 
    details: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Creates a new job record in the database."""
    db = SessionLocal()
    try:
        new_job = Job(
            id=job_id,
            status="queued",
            task_type=task_type,
            log_file=log_file,
            output_dir=output_dir,
            metrics_filename=metrics_filename,
            job_details=details or {},
        )
        db.add(new_job)
        db.commit()
        db.refresh(new_job)
        
        return {
            "job_id": new_job.id, 
            "status": new_job.status
        }
    finally:
        db.close()

def delete_job(job_id: str):
    """
    Deletes a job from the database.
    Returns the paths to clean up (log_file, output_dir) if successful.
    """
    db = SessionLocal()
    paths_to_delete = {}
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            paths_to_delete = {
                "log_file": job.log_file,
                "output_dir": job.output_dir
            }
            db.delete(job)
            db.commit()
            return paths_to_delete
        return None
    finally:
        db.close()

def update_job_status(job_id: str, status: str):
    """Updates the status of an existing job."""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = status
            db.commit()
    finally:
        db.close()

def update_job_pid(job_id: str, pid: int):
    """Updates the PID of a running job."""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.pid = pid
            db.commit()
    finally:
        db.close()

def read_log_file(job_id: str, tail_lines: Optional[int] = None) -> str:
    """
    Safely reads the log file content for a given job.
    If 'tail_lines' is specified, reads only the last N lines.
    """
    job = get_job(job_id)
    if not job or not job.log_file:
        return ""
        
    log_file_path = job.log_file
    if not os.path.exists(log_file_path):
        return ""
        
    try:
        with open(log_file_path, "r", encoding="utf-8", errors="replace") as f:
            if tail_lines is None:
                return f.read()
            else:
                # Use deque for an efficient "tail" operation
                last_lines = deque(f, maxlen=tail_lines)
                return "".join(last_lines)
    except Exception as e:
        return f"Error reading log file: {log_file_path}. Error: {e}"

# --- Dataset Functions ---

def create_dataset_entry(name: str, task_type: str, disk_path: str) -> RegisteredDataset:
    """Creates a new dataset record in the database."""
    db = SessionLocal()
    try:
        new_dataset = RegisteredDataset(
            name=name,
            task_type=task_type,
            status="processing",
            disk_path=disk_path
        )
        db.add(new_dataset)
        db.commit()
        db.refresh(new_dataset)
        return new_dataset
    finally:
        db.close()

def update_dataset_status(dataset_id: str, status: str, error_message: str = None):
    """Updates the status of an existing dataset."""
    db = SessionLocal()
    try:
        dataset = db.query(RegisteredDataset).filter(RegisteredDataset.id == dataset_id).first()
        if dataset:
            dataset.status = status
            dataset.error_message = error_message
            db.commit()
    finally:
        db.close()

def delete_dataset(dataset_id: str):
    """
    Deletes a dataset from the database.
    Returns the disk_path to clean up if successful.
    """
    db = SessionLocal()
    try:
        dataset = db.query(RegisteredDataset).filter(RegisteredDataset.id == dataset_id).first()
        if dataset:
            path_to_delete = dataset.disk_path
            db.delete(dataset)
            db.commit()
            return path_to_delete
        return None
    finally:
        db.close()

def get_dataset(dataset_id: str) -> Optional[RegisteredDataset]:
    """Retrieves a single dataset by its ID."""
    db = SessionLocal()
    try:
        return db.query(RegisteredDataset).filter(RegisteredDataset.id == dataset_id).first()
    finally:
        db.close()

def list_datasets_by_task(task_type: str) -> List[Dict[str, Any]]:
    """Lists all 'ready' datasets for a given task type."""
    db = SessionLocal()
    try:
        datasets = db.query(RegisteredDataset).filter(
            RegisteredDataset.task_type == task_type,
            RegisteredDataset.status == "ready"
        ).order_by(RegisteredDataset.name).all()
        return [{"id": d.id, "name": d.name} for d in datasets]
    finally:
        db.close()

# --- Security & Validation Functions ---

def get_valid_checkpoints_for_job(job: Job, ignore_status: bool = False) -> List[str]:
    """
    Scans a single job's output directory for valid checkpoints.
    - For Hugging Face jobs, it looks for 'checkpoint-*' subdirectories.
    - For YOLO jobs, it checks for 'weights/best.pt' and returns the
      main output_dir itself if found.
    """
    if not job:
        return []
        
    if not ignore_status and job.status != "completed":
        return []
    
    output_dir = job.output_dir
    if not os.path.isdir(output_dir):
        return []

    # Check for YOLO checkpoints first
    yolo_weights_path = os.path.join(output_dir, "weights", "best.pt")
    if os.path.exists(yolo_weights_path):
        # YOLO Model root output functions as the primary checkpoint
        return [output_dir]

    # --- Original Hugging Face checkpoint logic ---
    checkpoint_paths = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    valid_paths = [path for path in checkpoint_paths if os.path.isdir(path)]

    # Also check for final model in root output_dir
    # Hugging Face Trainer saves the final model directly in output_dir
    has_config = os.path.exists(os.path.join(output_dir, "config.json"))
    has_weights = os.path.exists(os.path.join(output_dir, "pytorch_model.bin")) or \
                  os.path.exists(os.path.join(output_dir, "model.safetensors"))
    
    if has_config and has_weights:
        valid_paths.append(output_dir)

    return valid_paths

def get_valid_checkpoints(task_type: str, run_name: Optional[str] = None) -> List[str]:
    """
    Gets a list of all valid, safe checkpoint paths for a given task,
    optionally filtered by run_name.
    """
    db = SessionLocal()
    try:
        query = db.query(Job).filter(
            Job.task_type == task_type,
            Job.status == "completed"
        )
        
        completed_jobs = query.all()
        
        all_checkpoints = []
        for job in completed_jobs:
            if run_name and job.job_details.get("run_name") != run_name:
                continue
                
            all_checkpoints.extend(get_valid_checkpoints_for_job(job))
            
        return all_checkpoints
    finally:
        db.close()

def is_valid_checkpoint_path(checkpoint_path: str) -> bool:
    """
    Security check: Verifies a given path is a valid, safe checkpoint path
    by checking it against the MODEL_OUTPUT_ROOT.
    """
    if not checkpoint_path:
        return False
        
    root_path = os.path.abspath(MODEL_OUTPUT_ROOT)
    full_path = os.path.abspath(checkpoint_path)
    
    if not full_path.startswith(root_path):
        return False 
        
        return False
        
    return True