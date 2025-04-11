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
    return time.strftime("%Y-%m-%d %H:%M:%S")

def save_metrics_to_csv(csv_path, model_name, metrics, used_hyperparams, val_file_path):
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
        