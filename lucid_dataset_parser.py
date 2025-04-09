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
import time
import random
import glob
import h5py
import numpy as np

from utils.constants import MAX_FLOW_LEN, TIME_WINDOW, SEED
from data.data_loader import count_packets_in_dataset
from data.args import get_dataset_parser, validate_args
from data.runner import parse_dataset_from_pcap, preprocess_dataset_from_data


# Sample commands
# split a pcap file into smaller chunks to leverage multi-core CPUs: tcpdump -r dataset.pcap -w dataset-chunk -C 1000
# dataset parsing (first step): python3 lucid_dataset_parser.py --dataset_type SYN2020 --dataset_folder ./sample-dataset/ --packets_per_flow 10 --dataset_id SYN2020 --traffic_type all --time_window 10
# dataset parsing (second step): python3 lucid_dataset_parser.py --preprocess_folder ./sample-dataset/

random.seed(SEED)
np.random.seed(SEED)

def main(argv):
    command_options = " ".join(str(x) for x in argv[1:])

    parser = get_dataset_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    if args.dataset_folder and args.dataset_type:
        parse_dataset_from_pcap(args, command_options)

    elif args.preprocess_folder or args.preprocess_file:
        preprocess_dataset_from_data(args, command_options)

    if args.balance_folder is not None and args.output_folder is not None:
        output_folder = args.output_folder[0] if args.output_folder is not None else args.balance_folder[0]
        datasets = []
        for folder in args.balance_folder:
            datasets += glob.glob(folder + '/*.hdf5')
        train_filelist = {}
        val_filelist = {}
        test_filelist = {}
        min_samples_train = float('inf')
        min_samples_val = float('inf')
        min_samples_test = float('inf')

        output_filename_prefix = None

        for file in datasets:
            filename = file.split('/')[-1].strip()
            dataset = h5py.File(file, "r")
            X = np.array(dataset["set_x"][:])  # features
            Y = np.array(dataset["set_y"][:])  # labels
            if 'train' in filename:
                key = filename.split('dataset')[0].strip() + 'dataset-balanced-train.hdf5'
                if output_filename_prefix ==None:
                    output_filename_prefix = filename.split('IDS')[0].strip()
                else:
                    if filename.split('IDS')[0].strip() != output_filename_prefix:
                        print ("Inconsistent datasets!")
                        exit()
                train_filelist[key] = (X,Y)
                if X.shape[0] < min_samples_train:
                    min_samples_train = X.shape[0]
            elif 'val' in filename:
                key = filename.split('dataset')[0].strip() + 'dataset-balanced-val.hdf5'
                if output_filename_prefix ==None:
                    output_filename_prefix = filename.split('IDS')[0].strip()
                else:
                    if filename.split('IDS')[0].strip() != output_filename_prefix:
                        print ("Inconsistent datasets!")
                        exit()
                val_filelist[key] = (X,Y)
                if X.shape[0] < min_samples_val:
                    min_samples_val = X.shape[0]
            elif 'test' in filename:
                key = filename.split('dataset')[0].strip() + 'dataset-balanced-test.hdf5'
                if output_filename_prefix ==None:
                    output_filename_prefix = filename.split('IDS')[0].strip()
                else:
                    if filename.split('IDS')[0].strip() != output_filename_prefix:
                        print ("Inconsistent datasets!")
                        exit()
                test_filelist[key] = (X, Y)
                if X.shape[0] < min_samples_test:
                    min_samples_test = X.shape[0]

        final_X = {'train':None,'val':None,'test':None}
        final_y = {'train':None,'val':None,'test':None}

        for key,value in train_filelist.items():
            X_short = value[0][:min_samples_train,...]
            y_short = value[1][:min_samples_train,...]

            if final_X['train'] is None:
                final_X['train'] = X_short
                final_y['train'] = y_short
            else:
                final_X['train'] = np.vstack((final_X['train'],X_short))
                final_y['train'] = np.hstack((final_y['train'],y_short))

        for key,value in val_filelist.items():
            X_short = value[0][:min_samples_val,...]
            y_short = value[1][:min_samples_val,...]

            if final_X['val'] is None:
                final_X['val'] = X_short
                final_y['val'] = y_short
            else:
                final_X['val'] = np.vstack((final_X['val'],X_short))
                final_y['val'] = np.hstack((final_y['val'],y_short))


        for key,value in test_filelist.items():
            X_short = value[0][:min_samples_test,...]
            y_short = value[1][:min_samples_test,...]

            if final_X['test'] is None:
                final_X['test'] = X_short
                final_y['test'] = y_short
            else:
                final_X['test'] = np.vstack((final_X['test'],X_short))
                final_y['test'] = np.hstack((final_y['test'],y_short))

        for key,value in final_X.items():
            filename = output_filename_prefix + 'IDS201X-dataset-balanced-' + key + '.hdf5'
            hf = h5py.File(output_folder + '/' + filename, 'w')
            hf.create_dataset('set_x', data=value)
            hf.create_dataset('set_y', data=final_y[key])
            hf.close()

        total_flows = final_y['train'].shape[0]+final_y['val'].shape[0]+final_y['test'].shape[0]
        ddos_flows = np.count_nonzero(final_y['train'])+np.count_nonzero(final_y['val'])+np.count_nonzero(final_y['test'])
        benign_flows = total_flows-ddos_flows
        [train_packets, val_packets, test_packets] = count_packets_in_dataset([final_X['train'], final_X['val'], final_X['test']])
        log_string = time.strftime("%Y-%m-%d %H:%M:%S") + " | total_flows (tot,ben,ddos):(" + str(total_flows) + "," + str(benign_flows) + "," + str(ddos_flows) + \
                     ") | Packets (train,val,test):(" + str(train_packets) + "," + str(val_packets) + "," + str(test_packets) + \
                     ") | Train/Val/Test sizes: (" + str(final_y['train'].shape[0]) + "," + str(final_y['val'].shape[0]) + "," + str(final_y['test'].shape[0]) + \
                     ") | options:" + command_options + " |\n"

        print(log_string)

        # saving log file
        with open(output_folder + '/history.log', "a") as myfile:
            myfile.write(log_string)


if __name__ == "__main__":
    main(sys.argv)
