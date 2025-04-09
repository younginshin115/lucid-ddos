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
import h5py
import glob

def load_dataset(path):
    """
    Load dataset from HDF5 file containing 'set_x' and 'set_y' datasets.

    Args:
        path (str): Glob pattern or filepath to HDF5 file

    Returns:
        Tuple[np.ndarray, np.ndarray]: X and Y datasets
    """
    filename = glob.glob(path)[0]
    dataset = h5py.File(filename, "r")
    set_x_orig = np.array(dataset["set_x"][:])
    set_y_orig = np.array(dataset["set_y"][:])

    X = np.reshape(set_x_orig, (set_x_orig.shape[0], set_x_orig.shape[1], set_x_orig.shape[2], 1))
    Y = set_y_orig
    return X, Y


def count_packets_in_dataset(X_list):
    """
    Count number of non-zero packets across all samples.

    Args:
        X_list (list of np.ndarray): List of 4D tensors

    Returns:
        list[int]: Packet counts per sample set
    """
    packet_counters = []
    for X in X_list:
        TOT = X.sum(axis=2)
        packet_counters.append(np.count_nonzero(TOT))
    return packet_counters


def all_same(items):
    """
    Check if all elements in a list are equal.

    Args:
        items (list): Any list

    Returns:
        bool: True if all values are the same
    """
    return all(x == items[0] for x in items)
