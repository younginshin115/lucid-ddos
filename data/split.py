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

"""
split.py

Provides functionality to split a flow-level dataset into training,
validation, and test sets based on desired ratio.
"""

import random
from utils.constants import TRAIN_SIZE
from collections import defaultdict

def train_test_split(flow_list, train_size=TRAIN_SIZE, shuffle=True):
    """
    Stratified split of flow samples into training and test sets based on class label.

    Args:
        flow_list (List[Tuple]): The list of flows (5-tuple, dict)
        train_size (float): Ratio of data to allocate for training (default: 0.9)
        shuffle (bool): Whether to shuffle before splitting

    Returns:
        Tuple:
            - train_set (List)
            - test_set (List)
    """
    label_to_flows = defaultdict(list)

    # Group flows by label
    for flow in flow_list:
        label = flow[1].get('label', 0)
        label_to_flows[label].append(flow)

    train_set, test_set = [], []

    for flows in label_to_flows.values():
        if shuffle:
            random.shuffle(flows)
        split_point = int(len(flows) * train_size)
        train_set.extend(flows[:split_point])
        test_set.extend(flows[split_point:])

    return train_set, test_set
