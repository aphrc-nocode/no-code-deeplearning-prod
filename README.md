# No-code deeplearning devops

This repo provides a production-ready, scalable pipeline for fine-tuning deeplearning models for multiple tasks, including Object Detection, Image Classification, and Image Segmentation.

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

### 1. Environment Setup
It is highly recommended to use a virtual environment. This project requires **Python 3.9 or higher**.

```bash
python3 -m venv .venv
source venv/bin/activate
```

### 2. Install Python Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

### 3. Start Infrastructure (Docker Compose - Recommended)

This system is containerized and requires 3 core background services (Nginx Proxy, FastAPI, Redis) and 4 dedicated GPU Celery Workers.

**Start all services seamlessly using Docker Compose:**

```bash
docker-compose up -d --build
```

You can view the auto-generated Swagger API documentation at `http://127.0.0.1:8000/docs` (or via your NGINX proxy domain). Use this interactive documentation to explore the available endpoints.

### 4. Start Infrastructure (Manual)

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
sudo apt-get update
sudo apt-get install redis-server -y
redis-server
```

**Step B: Start the FastAPI Server**
```bash
uvicorn fastapi_app:app --reload --port 8000 --host 0.0.0.0
```
The API documentation will be available at `http://127.0.0.0:8000/docs`.

**Step C: Start the Celery Worker**
```bash
celery -A celery_app.app worker --loglevel=info -c 1
```

---

## Scaling Celery Workers

The existing `docker-compose.yml` explicitly defines 4 Celery workers (`worker-0` through `worker-3`), each hard-pinned to individual physical GPUs on the host. 

To run fewer or more workers, simply edit the `docker-compose.yml` file to add or remove `worker-X` services, ensuring the `device_ids` match your available host NVIDIA layout.
