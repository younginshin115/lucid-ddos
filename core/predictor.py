import os
import csv
import glob
import time

from core.prediction_runner import run_prediction_loop_preprocessed
from data.data_loader import load_dataset, count_packets_in_dataset
from utils.constants import PREDICT_HEADER
from utils.path_utils import get_output_path
from utils.minmax_utils import static_min_max
from utils.prediction_utils import (
    load_model,
    extract_filename_prefix,
    warm_up_model,
    get_dataset_files,
    extract_model_metadata
)

def run_batch_prediction(args, output_folder):
    """
    Run batch prediction using one or more HDF5 datasets and models.

    Args:
        args (argparse.Namespace): Parsed command-line arguments
        output_folder (str): Folder path where predictions will be saved
    """
    # Create CSV file to store prediction results
    predict_file = open(
        get_output_path(output_folder, f"predictions-{time.strftime('%Y%m%d-%H%M%S')}.csv"),
        'a',
        newline=''
    )
    predict_file.truncate(0)  # Clear file content if it already exists
    predict_writer = csv.DictWriter(predict_file, fieldnames=PREDICT_HEADER)
    predict_writer.writeheader()
    predict_file.flush()

    iterations = args.iterations

    # Collect list of test dataset files
    dataset_filelist = get_dataset_files(args.predict)

    # Load one model or all models in the prediction folder
    model_list = [args.model] if args.model else glob.glob(args.predict + "/*.h5")

    for model_path in model_list:
        model_filename = os.path.basename(model_path).strip()

        # Extract prefix to match corresponding dataset files (e.g., '10t-10n-')
        filename_prefix = extract_filename_prefix(model_filename)

        # Load Keras model from .h5 file
        model = load_model(model_path)

        # Extract model parameters and normalization settings
        time_window, max_flow_len, model_name_string = extract_model_metadata(model_path)
        mins, maxs = static_min_max(time_window)

        # Perform a dummy forward pass to initialize GPU memory, etc.
        warm_up_model(model, sample_file=dataset_filelist[0])

        # Extract model name from filename (e.g., 'SYN2020-LUCID')
        model_name_string = model_filename.split(filename_prefix)[1].strip().split('.')[0].strip()

        for dataset_file in dataset_filelist:
            filename = os.path.basename(dataset_file).strip()

            # Only evaluate datasets matching the model's time-window and flow-length prefix
            if filename_prefix not in filename:
                continue

            # Load test dataset and count packets
            
            X, Y_true = load_dataset(dataset_file)

            for _ in range(iterations):
                run_prediction_loop_preprocessed(
                    X=X,
                    Y_true=Y_true,
                    model=model,
                    model_name=model_name_string,
                    source_name=os.path.basename(dataset_file),
                    writer=predict_writer
                )
            predict_file.flush()

    predict_file.close()

