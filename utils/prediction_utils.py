import os
import glob
from tensorflow.keras.models import load_model as keras_load_model
from data.data_loader import load_dataset

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
