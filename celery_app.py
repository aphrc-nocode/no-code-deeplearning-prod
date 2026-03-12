# celery_app.py

import os
import subprocess
import zipfile
import shutil
import sys
import glob
from celery import Celery
import job_store # Import our persistent job store

# --- Celery Configuration ---
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

app = Celery(
    "no_code_dl_worker",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["celery_app"] 
)

app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=1, # 1 task per worker at a time (for GPUs)
    task_acks_late=True,
)

@app.task(bind=True)
def run_training_task(self, job_id: str, cmd: list):
    """
    Celery task to run the training script, streaming stdout/stderr
    to the log file in real-time.
    """
    job_store.update_job_status(job_id, "running")
    
    job_data = job_store.get_job(job_id)
    log_file_path = job_data.log_file
    
    if not log_file_path:
        job_store.update_job_status(job_id, "failed")
        raise ValueError(f"No log file path found for job {job_id}")

    import traceback
    final_status = "failed" # Default to failed
    
    # Pre-open log file to write header details
    try:
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(f"=== Job {job_id} Started ===\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"CWD: {os.getcwd()}\n")
            f.write("==============================\n\n")
    except Exception as e:
        print(f"Failed to initialize log file {log_file_path}: {e}")
        # We can't proceed if we can't write logs (UI depends on it)
        job_store.update_job_status(job_id, "failed")
        return {"status": "failed", "job_id": job_id, "error": f"Log file init failed: {e}"}

    try:
        env = os.environ.copy()
        env["TQDM_DISABLE"] = "1" 
        env["PYTHONUNBUFFERED"] = "1" 
        
        # Use simple Popen first, logic handled below
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
            bufsize=1 
        )

        # --- CAPTURE PID ---
        try:
            job_store.update_job_pid(job_id, process.pid)
            print(f"Job {job_id} started with PID {process.pid}")
        except Exception as e:
            print(f"Failed to update PID for job {job_id}: {e}")
        # -------------------

        # --- Robust Logging Loop ---
        import select
        
        # Open in APPEND mode now, since we initialized it above
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            # Robust reader: read available bytes/chunks and flush frequently.
            # This captures \r-based progress updates and avoids waiting for '\n'.
            try:
                # Keep reading while the process is running
                while True:
                    # Wait up to 0.5 seconds for data to be available
                    ready, _, _ = select.select([process.stdout], [], [], 0.5)
                    if ready:
                        # Read a chunk of available text (not waiting for newline)
                        chunk = process.stdout.read(1024) # Increased chunk size
                        if not chunk:
                            # EOF reached
                            break

                        # Convert carriage-return-only progress updates into newline-separated chunks.
                        chunk = chunk.replace('\r', '\n')

                        log_file.write(chunk)
                        log_file.flush()
                    else:
                        # No data available right now. Check if process finished.
                        if process.poll() is not None:
                            # Read any remaining output
                            rem = process.stdout.read()
                            if rem:
                                rem = rem.replace('\r', '\n')
                                log_file.write(rem)
                                log_file.flush()
                            break
                        # otherwise loop and wait again
            except Exception as e:
                # Make sure we still capture exceptions in the worker log file
                log_file.write(f"\n[LOG STREAM ERROR] {e}\n")
                log_file.flush()
                print(f"Log stream error for {job_id}: {e}")

        # --- End Logging Loop ---
        
        process.stdout.close()
        return_code = process.wait() 
        
        if return_code == 0:
            # --- VERIFICATION: Ensure Final Model exists ---
            # Even if the script exited with 0, we must verify that the model was actually saved.
            # This prevents "phantom" completed jobs.
            job_data = job_store.get_job(job_id)
            valid_ckpts = job_store.get_valid_checkpoints_for_job(job_data, ignore_status=True)
            
            # For our strict "Final Model" policy, we generally expect the output_dir itself 
            # to be in the valid list (indicating a final model resides there).
            # However, simply checking if valid_ckpts is not empty is a good first step.
            # But let's be strict: if no valid checkpoints are found, it's a failure.
            if len(valid_ckpts) > 0:
                final_status = "completed"
            else:
                final_status = "failed"
                err_msg = "[PROCESS ENDED] Exit Code 0, but no valid model artifacts found. Marking as FAILED.\n"
                err_msg += "Possible causes: Output directory mismatch, write permissions, or script logic error."
                with open(log_file_path, "a") as f:
                    f.write(f"\n\n{err_msg}\n")
                print(f"Job {job_id}: {err_msg}")
            # --- END VERIFICATION ---
        else:
            final_status = "failed"
            # Append non-zero exit code info
            with open(log_file_path, "a") as f:
                f.write(f"\n\n[PROCESS ENDED] Exit Code: {return_code}\n")
            
    except Exception as e:
        final_status = "failed"
        err_msg = traceback.format_exc()
        try:
            with open(log_file_path, "a") as log_file:
                log_file.write(f"\n\n[CELERY WORKER ERROR]\n{err_msg}\n")
        except Exception:
            pass 
        print(f"Celery task exception: {err_msg}")
            
    finally:
        job_store.update_job_status(job_id, final_status)
        
    return {"status": final_status, "job_id": job_id}

# --- Data Root Extraction Utility ---
def find_data_root(start_path: str, task_type: str) -> str:
    """
    Scans the unzipped directory to find the actual root folder 
    containing the data (e.g., the folder that *contains* 'train').
    This is the most reliable signal for all dataset types.
    """
    
    # We are looking for the *parent* of the 'train' folder.
    for root, dirs, files in os.walk(start_path):
        if "train" in dirs:
            print(f"Found 'train' directory at: {os.path.join(root, 'train')}")
            # This 'root' is the data root.
            print(f"Setting data root to: {root}")
            return root

    # If no 'train' folder is found, we assume the user zipped
    # the contents directly, so the start_path *is* the root.
    print(f"Warning: Could not find a 'train' sub-directory. Using original path '{start_path}'.")
    return start_path
# --- End Utility ---


@app.task(bind=True)
def process_data_task(self, dataset_id: str, file_path: str, task_type: str):
    """
    Celery task to ingest an uploaded dataset.
    Handles ZIPs (unzip + preprocess) and single files (move to dest).
    """
    dataset = job_store.get_dataset(dataset_id)
    if not dataset:
        print(f"Error: Could not find dataset_id {dataset_id}")
        return

    processed_path = dataset.disk_path # Destination directory
    
    # Ensure dest dir exists
    os.makedirs(processed_path, exist_ok=True)
    
    temp_unzip_dir = f"{processed_path}_temp_unzip"
    
    try:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext != ".zip":
             raise ValueError(f"Task {task_type} requires .zip archives. Got {ext}")
             
        os.makedirs(temp_unzip_dir, exist_ok=True)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_unzip_dir)
            
            print(f"Unzipped data for {dataset_id} to {temp_unzip_dir}")

            # Use our new, robust data root finder
            data_root_for_script = find_data_root(temp_unzip_dir, task_type)
            
            if task_type == "object_detection":
                print("Running object detection preprocessing script...")
                script_path = "object_detection_utils/preprocess_data.py"
                cmd = [
                    sys.executable, script_path,
                    "--raw_data_dir", data_root_for_script, 
                    "--processed_data_dir", processed_path
                ]
                process = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print("Object detection preprocessing complete.")

            elif task_type in ["image_classification", "semantic_segmentation"]:
                if data_root_for_script != temp_unzip_dir:
                    print(f"Data root ({data_root_for_script}) is different from temp dir. Moving files.")
                    if os.path.exists(processed_path):
                        shutil.rmtree(processed_path)
                    shutil.move(data_root_for_script, processed_path)
                    print(f"Moved data from {data_root_for_script} to {processed_path}")
                else:
                    shutil.move(temp_unzip_dir, processed_path)
                    # Restore processed_path if it was the temp dir logic
                    if not os.path.exists(processed_path) and os.path.exists(temp_unzip_dir):
                         shutil.move(temp_unzip_dir, processed_path)
                    elif not os.path.exists(processed_path):
                         # If we moved temp_unzip_dir to processed_path, we are good.
                         pass
                    
                # Cleanup temp dir logic is complicated by the move. 
                # If we moved temp_unzip_dir, it's gone. If we moved a child, parent remains.
                # Simplest check in finally block handles this.

        job_store.update_dataset_status(dataset_id, "ready")
        
    except Exception as e:
        error_msg = str(e)
        if isinstance(e, subprocess.CalledProcessError):
            error_msg = f"STEP: {e.args}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
        print(f"Failed to process dataset {dataset_id}: {error_msg}")
        job_store.update_dataset_status(dataset_id, "failed", error_message=error_msg)
        
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(temp_unzip_dir):
            shutil.rmtree(temp_unzip_dir, ignore_errors=True)