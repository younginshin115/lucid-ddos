import os
import glob
import time
import csv
from tensorflow.keras.models import load_model as keras_load_model
from data.data_loader import load_dataset
from utils.path_utils import get_output_path
from utils.constants import PREDICT_HEADER

def load_model(model_path: str):
    """
    Load a Keras model from the given .h5 file path.

    Args:
        model_path (str): Full path to the .h5 model file

    Returns:
        keras.Model: Loaded Keras model
    """
    return keras_load_model(model_path)


def extract_filename_prefix(model_filename: str) -> str:
    """
    Extract prefix from model filename to match corresponding datasets.

    Example:
        '10t-10n-MyModel.h5' → '10t-10n-'

    Args:
        model_filename (str): Filename of the model

    Returns:
        str: Prefix to identify corresponding dataset files
    """
    parts = model_filename.split('-')
    return f"{parts[0].strip()}-{parts[1].strip()}-"


def warm_up_model(model, sample_file: str):
    """
    Perform a warm-up forward pass using the first sample to trigger GPU memory allocation.

    Args:
        model (keras.Model): Loaded Keras model
        sample_file (str): Path to an HDF5 file used for warm-up
    """
    X, _ = load_dataset(sample_file)
    _ = model.predict(X[:1], batch_size=1)


def get_dataset_files(dataset_folder: str) -> list:
    """
    Return a list of test dataset files (*.hdf5) in the given folder.

    Args:
        dataset_folder (str): Path to folder containing test datasets

    Returns:
        list: List of test file paths
    """
    return glob.glob(os.path.join(dataset_folder, "*test.hdf5"))

def extract_model_metadata(model_path: str):
    """
    Extract time_window, max_flow_len, and model name from the model filename.

    Args:
        model_path (str): Full path to model (.h5)

    Returns:
        tuple: (time_window: int, max_flow_len: int, model_name_string: str)
    """
    model_filename = os.path.basename(model_path)
    filename_prefix = extract_filename_prefix(model_filename)
    time_window = int(filename_prefix.split('t-')[0])
    max_flow_len = int(filename_prefix.split('t-')[1].split('n-')[0])
    model_name_string = model_filename.split(filename_prefix)[1].strip().split('.')[0].strip()
    return time_window, max_flow_len, model_name_string

def setup_prediction_output(output_folder: str) -> tuple:
    """
    Prepare CSV file and writer for saving prediction logs.

    Args:
        output_folder (str): Folder path where predictions will be saved

    Returns:
        tuple: (predict_file, csv_writer)
    """
    predict_file = open(
        get_output_path(output_folder, f"predictions-{time.strftime('%Y%m%d-%H%M%S')}.csv"),
        'a',
        newline=''
    )
    predict_file.truncate(0)
    predict_writer = csv.DictWriter(predict_file, fieldnames=PREDICT_HEADER)
    predict_writer.writeheader()
    predict_file.flush()
    return predict_file, predict_writer