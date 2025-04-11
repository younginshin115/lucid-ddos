import os
import csv
import time
import pyshark
import numpy as np
from tensorflow.keras.models import load_model

from data.data_loader import count_packets_in_dataset
from data.parser import parse_labels
from data.flow_utils import dataset_to_list_of_fragments
from data.live_process import process_live_traffic
from utils.constants import PREDICT_HEADER
from utils.preprocessing import normalize_and_padding
from utils.minmax_utils import static_min_max
from utils.path_utils import get_output_path
from utils.eval_logger import report_results


def run_live_prediction(args, output_folder: str):
    """
    Run live prediction on a network interface or pcap file.

    Args:
        args (argparse.Namespace): Parsed arguments containing model and interface/pcap
        output_folder (str): Path to save prediction logs
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

    if args.predict_live is None:
        print("Please specify a valid network interface or pcap file!")
        exit(-1)
    elif args.predict_live.endswith('.pcap'):
        cap = pyshark.FileCapture(args.predict_live)
        data_source = os.path.basename(args.predict_live)
    else:
        cap = pyshark.LiveCapture(interface=args.predict_live)
        data_source = args.predict_live

    print("Prediction on network traffic from:", data_source)

    # Load labels
    labels = parse_labels(args.dataset_type, args.attack_net, args.victim_net)

    # Load model
    if args.model is None or not args.model.endswith('.h5'):
        print("No valid model specified!")
        exit(-1)

    model_path = args.model
    model_filename = os.path.basename(model_path)
    filename_prefix = model_filename.split('n')[0] + 'n-'
    time_window = int(filename_prefix.split('t-')[0])
    max_flow_len = int(filename_prefix.split('t-')[1].split('n-')[0])
    model_name_string = model_filename.split(filename_prefix)[1].strip().split('.')[0].strip()

    model = load_model(model_path)
    mins, maxs = static_min_max(time_window)

    while True:
        samples = process_live_traffic(cap, args.dataset_type, labels, max_flow_len, traffic_type="all", time_window=time_window)

        if samples:
            X, Y_true, keys = dataset_to_list_of_fragments(samples)
            X = np.array(normalize_and_padding(X, mins, maxs, max_flow_len))
            Y_true = np.array(Y_true) if labels else None
            X = np.expand_dims(X, axis=3)

            pt0 = time.time()
            Y_pred = np.squeeze(model.predict(X, batch_size=2048) > 0.5, axis=1)
            pt1 = time.time()
            prediction_time = pt1 - pt0

            [packets] = count_packets_in_dataset([X])
            report_results(np.squeeze(Y_true), Y_pred, packets, model_name_string, data_source, prediction_time, predict_writer)
            predict_file.flush()

        elif isinstance(cap, pyshark.FileCapture):
            print("\nNo more packets in file", data_source)
            break

    predict_file.close()