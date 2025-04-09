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

import argparse

def get_usage_examples():
    return (
        "Usage examples:\n"
        "  python3 lucid_dataset_parser.py --dataset_type SYN2020 --dataset_folder ./sample-dataset/ "
        "--packets_per_flow 10 --dataset_id SYN2020 --traffic_type all --time_window 10\n"
        "  python3 lucid_dataset_parser.py --preprocess_folder ./sample-dataset/\n"
    )

def get_dataset_parser():
    parser = argparse.ArgumentParser(
        description=get_usage_examples(),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-d', '--dataset_folder', nargs='+', type=str,
                        help='Folder with the raw dataset (.pcap files)')

    parser.add_argument('-o', '--output_folder', nargs='+', type=str,
                        help='Directory where output files will be saved')

    parser.add_argument('-f', '--traffic_type', default='all', nargs='+', type=str,
                        help='Type of flow to process (all, benign, ddos)')

    parser.add_argument('-p', '--preprocess_folder', nargs='+', type=str,
                        help='Directory with preprocessed .data files')

    parser.add_argument('--preprocess_file', nargs='+', type=str,
                        help='Single preprocessed .data file to load')

    parser.add_argument('-b', '--balance_folder', nargs='+', type=str,
                        help='Folder(s) containing multiple .hdf5 datasets to balance and merge')

    parser.add_argument('-n', '--packets_per_flow', nargs='+', type=str,
                        help='Number of packets per flow sample (max flow length)')

    parser.add_argument('-s', '--samples', default=float('inf'), type=int,
                        help='Maximum number of training samples to keep (after balancing)')

    parser.add_argument('-i', '--dataset_id', nargs='+', type=str,
                        help='Identifier appended to output file names')

    parser.add_argument('-m', '--max_flows', default=0, type=int,
                        help='Maximum number of flows to process from pcap files (0 = all)')

    parser.add_argument('-l', '--label', default=1, type=int,
                        help='Label assigned to the DDoS class (default = 1)')

    parser.add_argument('-t', '--dataset_type', nargs='+', type=str,
                        help='Dataset type (e.g., DOS2017, DOS2018, DOS2019, SYN2020)')

    parser.add_argument('-w', '--time_window', nargs='+', type=str,
                        help='Duration of time window in seconds (e.g., 10)')

    parser.add_argument('--no_split', help='If set, do not split into train/val/test', action='store_true')

    parser.add_argument('--version', action='version', version='LUCID v1.0')

    return parser

def validate_args(args, parser):
    if not any([args.dataset_folder, args.preprocess_folder, args.preprocess_file, args.balance_folder]):
        parser.error("Please specify an input source.\n\n" + get_usage_examples())

    if args.dataset_folder and not args.dataset_type:
        parser.error("Please specify the dataset type (DOS2017, DOS2018, DOS2019, SYN2020) using the --dataset_type option.\n\n" + get_usage_examples())

    if args.balance_folder and not args.output_folder:
        parser.error("Please specify the output folder using the --output_folder option.\n\n" + get_usage_examples())
