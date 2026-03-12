# image_classification_utils/metrics.py

import evaluate
import numpy as np

def compute_metrics(eval_pred):
    """
    Computes accuracy, F1, precision, and recall for a given evaluation prediction.
    """
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    
    predictions = np.argmax(eval_pred.predictions, axis=1)
    references = eval_pred.label_ids
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=references)['accuracy']
    f1 = f1_metric.compute(predictions=predictions, references=references, average="weighted")['f1']
    precision = precision_metric.compute(predictions=predictions, references=references, average="weighted")['precision']
    recall = recall_metric.compute(predictions=predictions, references=references, average="weighted")['recall']
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
