import yaml
from pathlib import Path
import pandas as pd
from typing import Dict
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, classification_report

def compute_metrics(y_true: pd.Series, y_pred_proba: pd.Series, y_pred: pd.Series) -> Dict:
    metrics = {}
    metrics["roc_auc_score"] = roc_auc_score(y_true, y_pred_proba)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    metrics["accuracy_score"] = accuracy_score(y_true, y_pred)
    metrics["classification_report"] = classification_report(y_true, y_pred, output_dict=True)
    return metrics

def evaluate_performance(scores: pd.DataFrame, config: Dict) -> Dict:
    y_true = scores["y_true"]
    y_pred_proba = scores["y_pred_proba"]
    y_pred = scores["y_pred"]
    metrics = compute_metrics(y_true, y_pred_proba, y_pred)
    return metrics

def save_metrics(metrics: Dict, metrics_path: Path) -> None:
    with open(metrics_path, "w") as file:
        yaml.dump(metrics, file)
