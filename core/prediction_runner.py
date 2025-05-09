import numpy as np
import time
from data.data_loader import count_packets_in_dataset
from utils.preprocessing import normalize_and_padding
from utils.eval_logger import report_results


def run_prediction_loop(
    X_raw,
    Y_true,
    model,
    model_name,
    source_name,
    mins,
    maxs,
    max_flow_len,
    writer,
    packets=None
):
    """
    Normalize input, perform model inference, and log results.

    Args:
        X_raw (list or np.ndarray): Raw input features
        Y_true (list or np.ndarray or None): Ground-truth labels
        model (keras.Model): Loaded model to perform inference
        model_name (str): Model identifier for logging
        source_name (str): Name of the input source (e.g., filename or interface name)
        mins (list): Min values for normalization
        maxs (list): Max values for normalization
        max_flow_len (int): Flow padding length
        writer (csv.DictWriter): Writer for logging output
        packets (int, optional): Total number of packets in input. If None, auto-calculated.
    """
    # Normalize and shape the input
    X = np.array(normalize_and_padding(X_raw, mins, maxs, max_flow_len))
    X = np.expand_dims(X, axis=3)

    # Optional fallback to count packets
    if packets is None:
        [packets] = count_packets_in_dataset([X])

    # Model prediction
    pt0 = time.time()
    Y_pred = np.squeeze(model.predict(X, batch_size=2048) > 0.5, axis=1)
    pt1 = time.time()
    prediction_time = pt1 - pt0

    # Format labels if available
    if Y_true is not None:
        Y_true = np.squeeze(Y_true)

    # Write results
    report_results(
        Y_true,
        Y_pred,
        packets,
        model_name,
        source_name,
        prediction_time,
        writer
    )

def run_prediction_loop_preprocessed(
    X,
    Y_true,
    model,
    model_name,
    source_name,
    writer,
    packets=None
):
    """
    Perform prediction on preprocessed (normalized + padded) dataset.

    Args:
        X (np.ndarray): Preprocessed input data, shape (N, T, F)
        Y_true (np.ndarray or None): Ground-truth labels
        model (keras.Model): Keras model to perform inference
        model_name (str): Name of the model for logging
        source_name (str): Name of the dataset file
        writer (csv.DictWriter): CSV writer to log results
        packets (int, optional): Total packet count for evaluation
    """
    # Count packets if not provided
    if packets is None:
        [packets] = count_packets_in_dataset([X])

    # Run inference
    pt0 = time.time()
    Y_pred = np.squeeze(model.predict(X, batch_size=2048) > 0.5, axis=1)
    pt1 = time.time()
    prediction_time = pt1 - pt0

    # Format labels
    if Y_true is not None:
        Y_true = np.squeeze(Y_true)

    # Write results
    report_results(
        Y_true,
        Y_pred,
        packets,
        model_name,
        source_name,
        prediction_time,
        writer
    )
