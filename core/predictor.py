import os
import csv
import glob
import time
import numpy as np
from tensorflow.keras.models import load_model

from data.data_loader import load_dataset, count_packets_in_dataset
from utils.constants import PREDICT_HEADER
from utils.path_utils import get_output_path
from utils.eval_logger import report_results

def run_batch_prediction(args, output_folder):
    """
    Run batch prediction using one or more HDF5 datasets and models.

    Args:
        args (argparse.Namespace): Parsed command-line arguments
        output_folder (str): Folder path where predictions will be saved
    """
    # Create output CSV file for predictions
    predict_file = open(
        get_output_path(output_folder, f"predictions-{time.strftime('%Y%m%d-%H%M%S')}.csv"),
        'a',
        newline=''
    )
    predict_file.truncate(0)
    predict_writer = csv.DictWriter(predict_file, fieldnames=PREDICT_HEADER)
    predict_writer.writeheader()
    predict_file.flush()

    iterations = args.iterations
    dataset_filelist = glob.glob(args.predict + "/*test.hdf5")

    # Load either a specific model or all models in the folder
    model_list = [args.model] if args.model else glob.glob(args.predict + "/*.h5")

    for model_path in model_list:
        model_filename = os.path.basename(model_path).strip()
        filename_prefix = "-".join(model_filename.split('-')[:2]) + '-'
        model_name_string = model_filename.split(filename_prefix)[1].strip().split('.')[0].strip()
        model = load_model(model_path)

        # Warm up GPU with one forward pass
        warm_up_file = dataset_filelist[0]
        if filename_prefix in os.path.basename(warm_up_file):
            X, _ = load_dataset(warm_up_file)
            _ = model.predict(X[:1], batch_size=1)

        for dataset_file in dataset_filelist:
            filename = os.path.basename(dataset_file).strip()
            if filename_prefix not in filename:
                continue

            X, Y = load_dataset(dataset_file)
            [packets] = count_packets_in_dataset([X])
            Y_true = Y
            avg_time = 0

            for _ in range(iterations):
                pt0 = time.time()
                Y_pred = np.squeeze(model.predict(X, batch_size=2048) > 0.5)
                pt1 = time.time()
                avg_time += pt1 - pt0

            avg_time /= iterations

            report_results(np.squeeze(Y_true), Y_pred, packets, model_name_string, filename, avg_time, predict_writer)
            predict_file.flush()

    predict_file.close()
