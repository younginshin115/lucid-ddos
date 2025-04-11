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

import os 
from sklearn.utils import shuffle
from data.data_loader import load_dataset
from utils.constants import SEED

def parse_training_filename(filename):
    """
    Extract time window, max flow length, and dataset name from a training filename.

    Expected filename format: '10t-20n-DATASET.hdf5' or similar.

    Args:
        filename (str): The full path or just the filename (e.g., '10t-20n-SYN2020-train.hdf5')

    Returns:
        tuple: (time_window: int, max_flow_len: int, dataset_name: str)
    """
    parts = os.path.basename(filename).split('-')
    time_window = int(parts[0].replace('t', ''))
    max_flow_len = int(parts[1].replace('n', ''))
    dataset_name = parts[2]
    return time_window, max_flow_len, dataset_name

def load_and_shuffle_dataset(train_path, val_path):
    """
    Load and shuffle training and validation datasets using a fixed seed.

    Useful for consistent experiment results and reproducibility.

    Args:
        train_path (str): Glob pattern or full path to training HDF5 file
        val_path (str): Glob pattern or full path to validation HDF5 file

    Returns:
        tuple: ((X_train, Y_train), (X_val, Y_val)) — both shuffled
    """
    X_train, Y_train = load_dataset(train_path)
    X_val, Y_val = load_dataset(val_path)
    return (
        shuffle(X_train, Y_train, random_state=SEED),
        shuffle(X_val, Y_val, random_state=SEED)
    )