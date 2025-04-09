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
from data.flow_utils import count_flows


def train_test_split(flow_list, train_size=TRAIN_SIZE, shuffle=True):
    """
    Split a list of flow samples into training and test sets based on fragment count.

    Args:
        flow_list (List[Tuple]): The list of flows (5-tuple, dict)
        train_size (float): Ratio of data to allocate for training (default: 0.9)
        shuffle (bool): Whether to shuffle before splitting

    Returns:
        Tuple:
            - train_set (List)
            - test_set (List)
    """
    test_list = []
    _, (total_examples, _, _) = count_flows(flow_list)
    test_examples = total_examples - total_examples * train_size

    if shuffle:
        random.shuffle(flow_list)

    current_test_examples = 0
    while current_test_examples < test_examples:
        flow = flow_list.pop(0)
        test_list.append(flow)
        current_test_examples += len(flow[1]) - 1  # exclude 'label'

    return flow_list, test_list
