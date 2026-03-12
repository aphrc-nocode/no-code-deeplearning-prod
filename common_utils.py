# common_utils.py

import os
import json
from transformers import TrainerCallback

class JSONMetricsCallback(TrainerCallback):
    """
    A custom Hugging Face TrainerCallback that logs evaluation
    metrics to a JSONL file.
    
    This is used by all training scripts to populate the metrics
    file read by the API's /metrics/{job_id} endpoint.
    """
    def __init__(self, output_dir, metrics_filename="training_metrics.json"):
        super().__init__()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.metrics_file = os.path.join(output_dir, metrics_filename)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Called by the Trainer when logs are generated.
        We only write to the file if the logs contain 'eval_' keys.
        """
        if logs is not None:
            # Log all events (training loss, epoch updates) so UI shows progress
            # Previously only logged 'eval_' which missed intra-epoch progress
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(logs) + "\n")