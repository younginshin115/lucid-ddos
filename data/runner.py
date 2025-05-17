# Copyright (c) 2022 @ FBK - Fondazione Bruno Kessler
# Author: Roberto Doriguzzi-Corin
# Project: LUCID: A Practical, Lightweight Deep Learning Solution for DDoS Attack Detection
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import os, glob, time, pickle, h5py
from multiprocessing import Process, Manager

from data.parser import parse_labels, parse_labels_multiclass
from data.process_pcap import process_pcap
from data.flow_utils import count_flows, balance_dataset, dataset_to_list_of_fragments
from data.split import train_test_split
from data.data_loader import count_packets_in_dataset
from utils.preprocessing import normalize_and_padding
from utils.minmax_utils import static_min_max
from utils.constants import MAX_FLOW_LEN, TIME_WINDOW, TRAIN_SIZE
from utils.logging_utils import write_log, get_timestamp

def parse_dataset_from_pcap(args, command_options):
    """
    Parse and extract flows from .pcap files using multiprocessing.

    The resulting flows are saved as a .data pickle file and statistics are logged.

    Args:
        args (argparse.Namespace): Parsed CLI arguments
        command_options (str): Full string of command-line options (for logging)
    """
    manager = Manager()

    # Determine output folder
    if args.output_folder is not None and os.path.isdir(args.output_folder[0]):
        output_folder = args.output_folder[0]
    else:
        output_folder = args.dataset_folder[0]

    filelist = glob.glob(args.dataset_folder[0]+ '/*.pcap')

    # Load label definitions
    if args.label_mode == 'binary':
        in_labels = parse_labels(dataset_type=args.dataset_type[0], label=args.label)
    elif args.label_mode == 'multi':
        in_labels, label_map = parse_labels_multiclass(dataset_type=args.dataset_type[0])
    else:
        raise ValueError("Invalid label_mode. Must be 'binary' or 'multi'.")

    # Determine preprocessing parameters
    max_flow_len = int(args.packets_per_flow[0]) if args.packets_per_flow else MAX_FLOW_LEN
    time_window = float(args.time_window[0]) if args.time_window else TIME_WINDOW
    dataset_id = str(args.dataset_id[0]) if args.dataset_id else str(args.dataset_type[0])
    traffic_type = str(args.traffic_type[0]) if args.traffic_type else 'all'

    process_list = []
    flows_list = []

    start_time = time.time()

    # Start multiprocessed pcap parsing
    for file in filelist:
        flows = manager.list()
        p = Process(
            target=process_pcap,
            args=(file, args.dataset_type[0], in_labels, max_flow_len, flows, args.max_flows, traffic_type, time_window)
        )
        process_list.append(p)
        flows_list.append(flows)

    for p in process_list: p.start()
    for p in process_list: p.join()

    np.seterr(divide='ignore', invalid='ignore')

    try:
        preprocessed_flows = list(flows_list[0])
    except:
        print("ERROR: No traffic flows. Please check dataset folder and .pcap files.")
        exit(1)

    # Aggregate flows from all files
    for results in flows_list[1:]:
        preprocessed_flows += list(results)

    # Save flows as .data
    filename = f"{int(time_window)}t-{max_flow_len}n-{dataset_id}-preprocess"
    output_file = os.path.join(output_folder, filename).replace("//", "/")

    with open(output_file + '.data', 'wb') as filehandle:
        pickle.dump(preprocessed_flows, filehandle)

    # Count and log flow statistics
    (total_flows, ddos_flows, benign_flows),  (total_fragments, ddos_fragments, benign_fragments) = count_flows(preprocessed_flows)

    write_log([
        f"dataset_type:{args.dataset_type[0]}",
        f"flows (tot,ben,ddos):({total_flows},{benign_flows},{ddos_flows})",
        f"fragments (tot,ben,ddos):({total_fragments},{benign_fragments},{ddos_fragments})",
        f"options:{command_options}",
        f"process_time:{time.time() - start_time:.2f}"
    ], output_folder)


def preprocess_dataset_from_data(args, command_options):
    """
    Load .data files, balance the dataset, normalize flows, and split into train/val/test.

    Saves each split as an HDF5 file and logs statistics such as DDoS ratios and sample counts.

    Args:
        args (argparse.Namespace): Parsed CLI arguments
        command_options (str): Full command string used, for logging purposes
    """
    # Determine .data files and output folder
    if args.preprocess_folder:
        filelist = glob.glob(args.preprocess_folder[0] + '/*.data')
        output_folder = args.output_folder[0] if args.output_folder else args.preprocess_folder[0]
    else:
        filelist = args.preprocess_file
        output_folder = args.output_folder[0] if args.output_folder else os.path.dirname(os.path.realpath(filelist[0]))

    # Extract time_window, flow_len, dataset_id from filenames (consistency check)
    time_window, max_flow_len, dataset_id = None, None, None
    for file in filelist:
        parts = os.path.basename(file).split('-')
        current_time_window = int(parts[0].replace('t', ''))
        current_flow_len = int(parts[1].replace('n', ''))
        current_dataset_id = parts[2]

        if time_window and time_window != current_time_window:
            print("Inconsistent time windows!"); exit()
        if max_flow_len and max_flow_len != current_flow_len:
            print("Inconsistent flow lengths!"); exit()

        time_window = current_time_window
        max_flow_len = current_flow_len
        dataset_id = dataset_id if dataset_id and dataset_id == current_dataset_id else "IDS201X"

    # Load all .data files
    preprocessed_flows = []
    for file in filelist:
        with open(file, 'rb') as fh:
            preprocessed_flows += pickle.load(fh)

    # Balance and limit sample count
    preprocessed_flows, _, _ = balance_dataset(preprocessed_flows, args.samples)

    if not preprocessed_flows:
        print("Empty dataset after balancing.")
        exit()

    # Split into train, val, test
    train, test = train_test_split(preprocessed_flows, train_size=TRAIN_SIZE)
    train, val = train_test_split(train, train_size=TRAIN_SIZE)

    # Convert flows to 2D fragments and labels
    X_train, y_train, _ = dataset_to_list_of_fragments(train)
    X_val, y_val, _ = dataset_to_list_of_fragments(val)
    X_test, y_test, _ = dataset_to_list_of_fragments(test)

    # Normalize and pad inputs
    mins, maxs = static_min_max(time_window)
    norm_X_train = np.array(normalize_and_padding(X_train, mins, maxs, max_flow_len))
    norm_X_val = np.array(normalize_and_padding(X_val, mins, maxs, max_flow_len))
    norm_X_test = np.array(normalize_and_padding(X_test, mins, maxs, max_flow_len))

    y_train, y_val, y_test = np.array(y_train), np.array(y_val), np.array(y_test)

    # Save as HDF5
    filename_prefix = f"{time_window}t-{max_flow_len}n-{dataset_id}-dataset"
    for split, X, y in zip(["train", "val", "test"], [norm_X_train, norm_X_val, norm_X_test], [y_train, y_val, y_test]):
        with h5py.File(os.path.join(output_folder, f"{filename_prefix}-{split}.hdf5"), 'w') as hf:
            hf.create_dataset('set_x', data=X)
            hf.create_dataset('set_y', data=y)

    # Log final dataset summary
    # Count packets per split
    train_packets, val_packets, test_packets = count_packets_in_dataset([norm_X_train, norm_X_val, norm_X_test])

    # Compute example counts
    total_examples = len(y_train) + len(y_val) + len(y_test)
    total_ddos_examples = np.count_nonzero(np.concatenate([y_train, y_val, y_test]))
    total_benign_examples = total_examples - total_ddos_examples

    # Log in detailed format
    write_log([
        f"examples (tot,ben,ddos):({total_examples},{total_benign_examples},{total_ddos_examples})",
        f"Train/Val/Test sizes: ({len(y_train)},{len(y_val)},{len(y_test)})",
        f"Packets (train,val,test):({train_packets},{val_packets},{test_packets})",
        f"options: {command_options}"
    ], output_folder)

def merge_balanced_datasets(args, command_options):
    """
    Merge multiple balanced datasets (HDF5 format) into a unified dataset with consistent sample size.

    Ensures all splits (train, val, test) are truncated to the same minimum sample size and saves the result.

    Args:
        args (argparse.Namespace): Parsed CLI arguments
        command_options (str): Full command string used, for logging purposes
    """
    output_folder = args.output_folder[0] if args.output_folder else args.balance_folder[0]

    # Collect all dataset files
    datasets = []
    for folder in args.balance_folder:
        datasets += glob.glob(os.path.join(folder, '*.hdf5'))

    train_files, val_files, test_files = {}, {}, {}
    min_train, min_val, min_test = float('inf'), float('inf'), float('inf')
    output_prefix = None

    # Group datasets by split and determine the minimum sample count per split
    for file in datasets:
        filename = os.path.basename(file)
        with h5py.File(file, 'r') as f:
            X, Y = np.array(f["set_x"][:]), np.array(f["set_y"][:])
        if 'train' in filename:
            key = filename.split('dataset')[0] + 'dataset-balanced-train.hdf5'
            output_prefix = output_prefix or filename.split('IDS')[0].strip()
            if filename.split('IDS')[0].strip() != output_prefix:
                print("Inconsistent datasets!"); exit()
            train_files[key] = (X, Y)
            min_train = min(min_train, X.shape[0])
        elif 'val' in filename:
            key = filename.split('dataset')[0] + 'dataset-balanced-val.hdf5'
            output_prefix = output_prefix or filename.split('IDS')[0].strip()
            if filename.split('IDS')[0].strip() != output_prefix:
                print("Inconsistent datasets!"); exit()
            val_files[key] = (X, Y)
            min_val = min(min_val, X.shape[0])
        elif 'test' in filename:
            key = filename.split('dataset')[0] + 'dataset-balanced-test.hdf5'
            output_prefix = output_prefix or filename.split('IDS')[0].strip()
            if filename.split('IDS')[0].strip() != output_prefix:
                print("Inconsistent datasets!"); exit()
            test_files[key] = (X, Y)
            min_test = min(min_test, X.shape[0])

    final_X, final_y = {'train': None, 'val': None, 'test': None}, {'train': None, 'val': None, 'test': None}

    # Trim all datasets to the same minimum size per split and stack together
    for split, file_dict, min_samples in [('train', train_files, min_train), ('val', val_files, min_val), ('test', test_files, min_test)]:
        for _, (X, Y) in file_dict.items():
            X_short = X[:min_samples]
            Y_short = Y[:min_samples]
            final_X[split] = X_short if final_X[split] is None else np.vstack((final_X[split], X_short))
            final_y[split] = Y_short if final_y[split] is None else np.hstack((final_y[split], Y_short))

    # Save the final merged datasets
    for split in ['train', 'val', 'test']:
        filename = f"{output_prefix}IDS201X-dataset-balanced-{split}.hdf5"
        with h5py.File(os.path.join(output_folder, filename), 'w') as hf:
            hf.create_dataset('set_x', data=final_X[split])
            hf.create_dataset('set_y', data=final_y[split])

    # Count and log final statistics
    total_flows = sum(final_y[split].shape[0] for split in ['train', 'val', 'test'])
    ddos_flows = sum(np.count_nonzero(final_y[split]) for split in ['train', 'val', 'test'])
    benign_flows = total_flows - ddos_flows
    [train_packets, val_packets, test_packets] = count_packets_in_dataset(
        [final_X['train'], final_X['val'], final_X['test']]
    )
    
    write_log([
        f"total_flows (tot,ben,ddos):({total_flows},{benign_flows},{ddos_flows})",
        f"Packets (train,val,test):({train_packets},{val_packets},{test_packets})",
        f"Train/Val/Test sizes: ({final_y['train'].shape[0]},{final_y['val'].shape[0]},{final_y['test'].shape[0]})",
        f"options: {command_options}"
    ], output_folder)


