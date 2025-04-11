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

#Sample commands
# Training: python3 lucid_cnn.py --train ./sample-dataset/  --epochs 100 -cv 5
# Testing: python3  lucid_cnn.py --predict ./sample-dataset/ --model ./sample-dataset/10t-10n-SYN2020-LUCID.h5

import argparse
from utils.constants import DEFAULT_EPOCHS

def get_usage_examples():
    """
    Returns example usage strings for the CLI.

    This is shown in the help message to guide users on how to use the script.

    Returns:
        str: A string containing example commands.
    """
    return (
        "Usage examples:\n"
        "  python3 lucid_cnn.py --train ./datasets/SYN2020 -e 30\n"
        "  python3 lucid_cnn.py --predict ./datasets/SYN2020 --model ./models/best_model.h5\n"
        "  python3 lucid_cnn.py --predict_live eth0 --dataset_type SYN2020 --model ./models/best_model.h5\n"
    )

def get_lucid_cnn_parser():
    """
    Create and configure the argument parser for lucid_cnn.py.

    Returns:
        argparse.ArgumentParser: The configured argument parser object.
    """
    parser = argparse.ArgumentParser(
        description=get_usage_examples(),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-t', '--train', nargs='+', type=str,
                        help='Folder with training dataset(s) (e.g., .hdf5)')

    parser.add_argument('-e', '--epochs', default=DEFAULT_EPOCHS, type=int,
                        help='Number of training epochs')

    parser.add_argument('-cv', '--cross_validation', default=0, type=int,
                        help='Number of folds for cross-validation (0 = no CV)')

    parser.add_argument('-a', '--attack_net', default=None, type=str,
                        help='Attacker subnet (used for label generation in live traffic)')

    parser.add_argument('-v', '--victim_net', default=None, type=str,
                        help='Victim subnet (used for label generation in live traffic)')

    parser.add_argument('-p', '--predict', nargs='?', type=str,
                        help='Path to folder with test datasets for batch prediction')

    parser.add_argument('-pl', '--predict_live', nargs='?', type=str,
                        help='Live interface or .pcap file for real-time prediction')

    parser.add_argument('-i', '--iterations', default=1, type=int,
                        help='Number of prediction iterations (averaged)')

    parser.add_argument('-m', '--model', type=str,
                        help='Path to trained .h5 model file')

    parser.add_argument('-y', '--dataset_type', default=None, type=str,
                        help='Dataset type for label parsing (e.g., DOS2017, SYN2020)')

    parser.add_argument('--version', action='version', version='LUCID CNN v1.0')

    return parser

def get_args():
    """
    Parse and validate command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments after validation.
    """
    parser = get_lucid_cnn_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    return args

def validate_args(args, parser):
    """
    Perform logical validation on argument combinations.

    This ensures mutually exclusive options are not used together and that required options
    are provided in certain modes (e.g., predict_live requires --model and --dataset_type).

    Args:
        args (argparse.Namespace): The parsed arguments to validate
        parser (argparse.ArgumentParser): The parser object (used for error reporting)
    """
    usage = get_usage_examples()

    if not any([args.train, args.predict, args.predict_live]):
        parser.error("Please specify one of the following modes: --train, --predict, or --predict_live.\n\n" + usage)

    if args.train and args.predict:
        parser.error("Cannot use --train and --predict at the same time.\n\n" + usage)

    if args.train and args.predict_live:
        parser.error("Cannot use --train and --predict_live at the same time.\n\n" + usage)

    if args.predict and not args.model:
        print("No model specified for --predict. All .h5 files in the folder will be used.")

    if args.predict_live:
        if not args.model or not args.model.endswith(".h5"):
            parser.error("Please specify a valid .h5 model file using --model when using --predict_live.\n\n" + usage)
        if not args.dataset_type:
            parser.error("Please specify the dataset type (e.g., DOS2017, SYN2020) using --dataset_type when using --predict_live.\n\n" + usage)
