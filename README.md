# No-code deeplearning devops

This repo provides a production-ready, scalable pipeline for fine-tuning deeplearning models for multiple tasks, including Object Detection, Image Classification, and Image Segmentation.

The system is built on a modern, decoupled architecture:
* **Frontend:** An **R Shiny** application (`ui.R`, `server.R`) provides a low-code UI for managing data and experiments.
* **Backend API:** A **FastAPI** server (`fastapi_app.py`) handles all web requests, manages job state, and serves results.
* **Job Queue:** **Redis** acts as a high-speed message broker to queue training jobs.
* **Compute Workers:** **Celery** workers (`celery_app.py`) run as separate, scalable processes that consume jobs from the queue and execute the training scripts.
* **Persistence:** A **SQLite** database (`job_store.py`) stores all job and dataset metadata.

## Project Structure

```

no-code-deeplearning/
├── app_data/                # (Git-ignored) Default persistent storage for datasets
├── model_outputs/           # (Git-ignored) Default storage for trained models
├── object_detection_utils/
├── image_classification_utils/
├── image_segmentation_utils/
├── ...
├── object_detection_train.py
├── image_classification_train.py
├── image_segmentation_train.py
├── ...
├── fastapi_app.py           # The API Server
├── celery_app.py            # The Celery Worker definition
├── job_store.py             # Database models and logic
├── model_registry.py        # Single source of truth for supported models
├── common_utils.py          # Shared utilities (e.g., JSONMetricsCallback)
├── ...
├── ui.R                     # Shiny UI definition
├── server.R                 # Shiny server logic
└── requirements.txt

```

---

## Setup and Installation

### 1. Environment Setup
It is highly recommended to use a virtual environment. This project requires **Python 3.9 or higher**.

```bash
python3 -m venv venv
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

You can view the auto-generated API documentation at `http://127.0.0.1:8000/docs` (or via your NGINX proxy domain).

### 4. Start Infrastructure (Manual)

If you prefer to run the services locally for development instead of using Docker, you will need multiple terminal windows. Ensure your virtual environment is activated in each Python terminal.

**1. Start Redis**
Ensure Redis is installed on your system.
```bash
sudo apt-get update
sudo apt-get install redis-server -y
redis-server
redis-cli ping
```

**2. Start the FastAPI Server**
```bash
uvicorn fastapi_app:app --reload --port 8000 --host 0.0.0.0
```
The API documentation will be available at `http://127.0.0.0:8000/docs`.

**3. Start the Celery Worker**
```bash
celery -A celery_app worker --loglevel=info
```

---

## How to Use the Platform

**1. Run the Shiny App:**
Open the `ui.R` or `server.R` file in RStudio and click **"Run App"**.

**2. Upload Data:**

  * Go to the **"Data Management"** tab.
  * Give your dataset a name (e.g., "oxford-pets-seg").
  * Select the correct **"Task Type"** (this is critical).
  * Upload your `.zip` file.
      * **Note:** Your zip file must contain the expected data structure for that task (e.g., `train/`, `validation/` folders). For segmentation, it must also include a `metadata.json` file.
  * Click **"Upload and Process Dataset"**. The UI will show a "processing" status and will auto-refresh the "Registered Datasets" table when complete.

**3. Configure a Job:**

  * Go to the **"Task Configuration"** sidebar.
  * Select your desired task (e.g., "Image Segmentation").
  * From the "Paths & Naming" panel, select your newly uploaded dataset from the "Select Dataset" dropdown.
  * Select a model architecture and checkpoint from the registry.
  * Set a `run_name` and configure all other hyperparameters.

**4. Start Training:**

  * Click the "Start Image Segmentation Job" button.
  * The UI will automatically switch to the **"Live Training"** tab.

**5. Monitor Training:**

  * On the **"Live Training"** tab, you can watch the job status change from `queued` to `running`.
  * You can view the **"Full Log"** in real-time (now cleaned of `tqdm` artifacts).
  * You can view the **"Metrics Table"** as evaluation epochs complete.
  * You can use the **"Live Metric Visualization"** dropdown to plot any metric from the table.

**6. Review History & Run Inference:**

  * Use the **"Training History"** tab to review, filter, and load metrics from all past jobs.
  * Use the **"Inference"** tab to select a task, find your trained checkpoints by `run_name`, and test them on new images or audio files.

## Scaling (Docker Compose)

The existing `docker-compose.yml` explicitly defines 4 Celery workers (`worker-0` through `worker-3`), each hard-pinned to individual physical GPUs on the host. 

To run fewer or more workers, simply edit the `docker-compose.yml` file to add or remove `worker-X` services, ensuring the `device_ids` match your available host NVIDIA layout.
