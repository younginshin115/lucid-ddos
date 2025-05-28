import time, os, csv, json
from utils.constants import VAL_HEADER
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def write_log(parts: list[str], output_folder: str, include_timestamp: bool = True):
    """
    Joins log parts, prepends timestamp (optional), prints and saves to history.log.

    Args:
        parts (list[str]): List of log segments (e.g., examples, sizes, packets, etc.)
        output_folder (str): Where to write history.log
        include_timestamp (bool): Whether to add timestamp as prefix
    """
    message = " | ".join(parts) + " |"
    if include_timestamp:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        message = f"{timestamp} | {message}"
    print(message)
    with open(os.path.join(output_folder, 'history.log'), 'a') as f:
        f.write(message + '\n')

def get_timestamp():
    """
    Get the current timestamp in a human-readable format.

    Returns:
        str: Timestamp string formatted as 'YYYY-MM-DD HH:MM:SS'
    """
    return time.strftime("%Y-%m-%d %H:%M:%S")

def save_evaluation_artifacts(base_path, model_name, metrics, used_hyperparams, val_file_path, label_mode, class_labels=None):
    """
    Save evaluation results: CSV, confusion matrix JSON, and confusion matrix image.

    Args:
        base_path (str): Base path without extension (e.g., '/models/SYN2020-LUCID')
        model_name (str): Name of the model
        metrics (dict): Dict with accuracy, f1, precision, recall, confusion_matrix, samples
        used_hyperparams (dict): Hyperparameters used
        val_file_path (str): Path to the validation set
        label_mode (str): 'binary' or 'multi'
        class_labels (list[str] or list[int], optional): Class names for confusion matrix image
    """
    
    # 1. Save metrics to CSV
    csv_path = base_path + ".csv"
    with open(csv_path, 'w', newline='') as val_file:
        val_writer = csv.DictWriter(val_file, fieldnames=VAL_HEADER)
        val_writer.writeheader()
        val_writer.writerow({
            'Model': model_name,
            'Samples': metrics['samples'],
            'Accuracy': f"{metrics['accuracy']:05.4f}",
            'F1Score': f"{metrics['f1']:05.4f}",
            'Precision': f"{metrics['precision']:05.4f}",
            'Recall': f"{metrics['recall']:05.4f}",
            'Hyper-parameters': used_hyperparams,
            'Validation Set': val_file_path,
            'Label Mode': label_mode
        })

    # 2. Save confusion matrix to JSON
    matrix_json_path = base_path + "_confusion_matrix.json"
    with open(matrix_json_path, "w") as f:
        json.dump(metrics["confusion_matrix"], f, indent=2)
    
    # 3. Save confusion matrix image
    png_path = base_path + "_confusion_matrix.png"
    if class_labels is None:
        class_labels = list(range(len(metrics["confusion_matrix"])))
    save_confusion_matrix_image(metrics["confusion_matrix"], class_labels, png_path)

def save_confusion_matrix_image(conf_matrix, class_labels, output_path, title="Confusion Matrix"):
    """
    Save confusion matrix as a heatmap image.

    Args:
        conf_matrix (list[list[int]]): Confusion matrix as a nested list (or np.ndarray)
        class_labels (list[str] or None): List of class names for axis ticks
        output_path (str): Path to save the output PNG image
        title (str): Title for the heatmap
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        np.array(conf_matrix), 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=class_labels,
        yticklabels=class_labels
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()