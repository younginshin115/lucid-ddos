from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import os
from datetime import datetime

def create_tensorboard_callback(base_log_dir="logs/tensorboard", experiment_name=None):
    """
    Create a TensorBoard callback with a timestamped directory.

    Args:
        base_log_dir (str): Base directory for logs.
        experiment_name (str): Optional experiment name.

    Returns:
        TensorBoard: Configured TensorBoard callback
        str: Path to the log directory
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(base_log_dir, f"{timestamp}_{experiment_name}" if experiment_name else timestamp)
    os.makedirs(log_dir, exist_ok=True)
    return TensorBoard(log_dir=log_dir)

def create_early_stopping_callback(patience=10, monitor='val_loss'):
    """
    Create an EarlyStopping callback.

    Args:
        patience (int): Number of epochs with no improvement before stopping
        monitor (str): Metric to monitor

    Returns:
        EarlyStopping: Configured EarlyStopping callback
    """
    return EarlyStopping(monitor=monitor, mode='min', verbose=1, patience=patience)

def create_model_checkpoint_callback(model_path, monitor='val_accuracy'):
    """
    Create a ModelCheckpoint callback.

    Args:
        model_path (str): Path to save the best model
        monitor (str): Metric to monitor

    Returns:
        ModelCheckpoint: Configured ModelCheckpoint callback
    """
    return ModelCheckpoint(
        filepath=model_path + ".h5",
        monitor=monitor,
        mode='max',
        verbose=1,
        save_best_only=True
    )
