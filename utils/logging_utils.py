import time, os, csv
from utils.constants import VAL_HEADER

def write_log(message: str, output_folder: str, prefix: str = ''):
    """
    Prints and appends a message to the history.log file in the given folder.

    Args:
        message (str): The log message to print and write
        output_folder (str): Path to the folder where history.log should be saved
        prefix (str): Optional prefix (like timestamp)
    """
    if prefix:
        message = f"{prefix} | {message}"
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
