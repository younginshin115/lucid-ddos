import os
import csv
import glob
import time
import numpy as np

from data.data_loader import load_dataset, count_packets_in_dataset
from utils.constants import PREDICT_HEADER
from utils.path_utils import get_output_path
from utils.eval_logger import report_results
from utils.prediction_utils import (
    load_model,
    extract_filename_prefix,
    warm_up_model,
    get_dataset_files
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
            X, Y = load_dataset(dataset_file)
            [packets] = count_packets_in_dataset([X])
            Y_true = Y
            avg_time = 0

            # Repeat prediction multiple times to average runtime
            for _ in range(iterations):
                pt0 = time.time()
                Y_pred = np.squeeze(model.predict(X, batch_size=2048) > 0.5)
                pt1 = time.time()
                avg_time += pt1 - pt0

            avg_time /= iterations

            # Report and log results
            report_results(
                np.squeeze(Y_true),
                Y_pred,
                packets,
                model_name_string,
                filename,
                avg_time,
                predict_writer
            )
            predict_file.flush()

    predict_file.close()

