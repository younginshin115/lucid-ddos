import time, os, csv
from utils.constants import VAL_HEADER
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime

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


def save_metrics_to_csv(csv_path, model_name, metrics, used_hyperparams, val_file_path):
    """
    Save evaluation metrics to a CSV file for record-keeping and analysis.

    This writes a single row into a CSV file with model name, accuracy, F1 score,
    number of samples, used hyperparameters, and validation set information.

    Args:
        csv_path (str): Full path to the CSV file to write
        model_name (str): Name of the model (e.g., 'SYN2020-LUCID')
        metrics (dict): Dictionary with keys 'accuracy', 'f1', 'samples'
        used_hyperparams (dict): Dictionary of hyperparameters used for training
        val_file_path (str): Path to the validation dataset used
    """
    with open(csv_path, 'w', newline='') as val_file:
        val_writer = csv.DictWriter(val_file, fieldnames=VAL_HEADER)
        val_writer.writeheader()
        val_writer.writerow({
            'Model': model_name,
            'Samples': metrics['samples'],
            'Accuracy': f"{metrics['accuracy']:05.4f}",
            'F1Score': f"{metrics['f1']:05.4f}",
            'Hyper-parameters': used_hyperparams,
            'Validation Set': val_file_path
        })

def create_tensorboard_callback(base_log_dir="logs/tensorboard", experiment_name=None):
    """
    Create a TensorBoard callback for visualizing training in real-time.

    Args:
        base_log_dir (str): Base path to store logs.
        experiment_name (str): Optional experiment name for easier tracking.

    Returns:
        TensorBoard: TensorBoard callback.
        str: The path to the log directory.
    """

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if experiment_name:
        log_dir = os.path.join(base_log_dir, f"{timestamp}_{experiment_name}") if experiment_name else os.path.join(base_log_dir, timestamp)

    os.makedirs(log_dir, exist_ok=True)
    return TensorBoard(log_dir=log_dir)