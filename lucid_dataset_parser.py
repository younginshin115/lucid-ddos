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

import sys
import random
import numpy as np

from utils.constants import SEED
from data.args import get_dataset_parser, validate_args
from data.runner import parse_dataset_from_pcap, preprocess_dataset_from_data, merge_balanced_datasets

# Sample commands
# split a pcap file into smaller chunks to leverage multi-core CPUs: tcpdump -r dataset.pcap -w dataset-chunk -C 1000
# dataset parsing (first step): python3 lucid_dataset_parser.py --dataset_type SYN2020 --dataset_folder ./sample-dataset/ --packets_per_flow 10 --dataset_id SYN2020 --traffic_type all --time_window 10
# dataset parsing (second step): python3 lucid_dataset_parser.py --preprocess_folder ./sample-dataset/

random.seed(SEED)
np.random.seed(SEED)

def main(argv):
    parser = get_dataset_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    command_options = " ".join(str(x) for x in argv[1:])

    if args.dataset_folder and args.dataset_type:
        parse_dataset_from_pcap(args, command_options)

    elif args.preprocess_folder or args.preprocess_file:
        preprocess_dataset_from_data(args, command_options)
    
    elif args.balance_folder:
        merge_balanced_datasets(args, command_options)

if __name__ == "__main__":
    main(sys.argv)
