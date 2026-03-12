# No-Code Deep Learning API Service

This project provides a production-ready, scalable REST API for fine-tuning and running inference on deep learning models across multiple tasks, including Object Detection, Image Classification, and Image Segmentation.

This service acts as an independent headless backend, fully decoupled from any specific front-end application. It is designed to be consumed by client applications, dashboards, or automation scripts.

The system is built on a modern, asynchronous architecture:
* **Backend API:** A **FastAPI** server (`fastapi_app.py`) handles all HTTP REST requests, manages job state, and serves results.
* **Job Queue:** **Redis** acts as a high-speed message broker to queue asynchronous training and inference jobs.
* **Compute Workers:** **Celery** workers (`celery_app.py`) run as separate, scalable processes that consume jobs from the queue and execute the heavy-lifting deep learning scripts.
* **Persistence:** A **SQLite** database (`job_store.py`) stores all job metadata, configurations, and dataset tracking.

---

## Project Structure

```text
no-code-deeplearning-prod/
├── app_data/                # (Git-ignored) Default persistent storage for uploaded datasets
├── model_outputs/           # (Git-ignored) Default storage for trained model weights & logs
├── runs/                    # (Git-ignored) Training run artifacts
├── object_detection_utils/  # Task-specific utility modules
├── image_classification_utils/
├── image_segmentation_utils/
├── object_detection_train.py       # Training scripts (HuggingFace/YOLO)
├── image_classification_train.py
├── image_segmentation_train.py
├── *_inference.py           # Inference scripts for predicting on new data
├── fastapi_app.py           # The FastAPI Web Server Entrypoint
├── celery_app.py            # The Celery Worker definition and task routing
├── job_store.py             # SQLite database models and logic
├── model_registry.py        # Central registry for supported foundation models
├── common_utils.py          # Shared utilities (e.g., Metrics parsing)
├── docker-compose.yml       # Production deployment configuration
├── Dockerfile               # Container build instructions
├── nginx.conf               # Reverse proxy configuration
└── requirements.txt         # Python dependencies
```

---

## Setup and Installation

### 1. Start Infrastructure (Docker Compose - Recommended)

This system is containerized and requires 3 core background services (Nginx Proxy, FastAPI, Redis) and 4 dedicated GPU Celery Workers.

**Start all services seamlessly using Docker Compose:**

```bash
docker-compose up -d --build
```

You can view the auto-generated Swagger API documentation at `http://127.0.0.1:8000/docs` (or via your NGINX proxy domain). Use this interactive documentation to explore the available endpoints.

### 2. Start Infrastructure (Local / Manual)

If you prefer to run the services locally for development instead of using Docker, you will need multiple terminal windows.

**Prerequisites:** 
- Python 3.9+
- A virtual environment is highly recommended.
- Redis server installed on your host system.

```bash
# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ensure your virtual environment is activated in each Python terminal for the following steps:

**Step A: Start Redis**
```bash
redis-server
```

**Step B: Start the FastAPI Server**
```bash
uvicorn fastapi_app:app --host 127.0.0.1 --port 8000 --reload
```
The API documentation will be available at `http://127.0.0.1:8000/docs`.

**Step C: Start the Celery Worker**
```bash
celery -A celery_app worker --loglevel=info
```

---

## Scaling Celery Workers

The existing `docker-compose.yml` explicitly defines 4 Celery workers (`worker-0` through `worker-3`), each hard-pinned to individual physical GPUs on the host. 

To run fewer or more workers, simply edit the `docker-compose.yml` file to add or remove `worker-X` services, ensuring the `device_ids` match your available host NVIDIA GPU layout.

---

## Core API Workflows

This service exposes several endpoints for managing the deep learning pipeline. Please check the `/docs` endpoint for the complete schema and interactive testing.

### 1. Data Management
- `POST /datasets/upload`: Upload zip files containing formatted datasets for specific tasks.
- `GET /datasets/list`: Retrieve a list of registered datasets.

### 2. Model Registry
- `GET /registry`: Fetch the list of supported foundation models and their expected tasks.

### 3. Training Jobs
- `POST /jobs/submit`: Submit a new asynchronous training job to the Celery queue.
- `GET /jobs/list`: List all historical and active training jobs.
- `GET /metrics/{job_id}`: Stream or fetch training metrics for a specific job.
- `POST /jobs/cancel/{job_id}`: Abort an active training task.

### 4. Inference
- `GET /checkpoints`: List available trained model checkpoints.
- `POST /inference/{task_type}`: Submit an image for prediction using a specific trained model.

### 5. System Health
- `GET /system/health`: Monitor GPU, CPU, RAM, and Disk utilization across the host machine.