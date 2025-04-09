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
flow_utils.py

Utility functions for analyzing and transforming flow-level data
for training and evaluation of DDoS detection models.

Author: Roberto Doriguzzi-Corin
License: Apache 2.0
"""

import random

def count_flows(preprocessed_flows):
    """
    Count the number of total, DDoS, and benign flows and fragments.

    Args:
        preprocessed_flows (List[Tuple[5-tuple, Dict]]): Flow data with labels

    Returns:
        Tuple:
            - (total_flows, ddos_flows, benign_flows)
            - (total_fragments, ddos_fragments, benign_fragments)
    """
    ddos_flows = 0
    total_flows = len(preprocessed_flows)
    ddos_fragments = 0
    total_fragments = 0

    for _, flow in preprocessed_flows:
        flow_fragments = len(flow) - 1
        total_fragments += flow_fragments
        if flow.get('label', 0) > 0:
            ddos_flows += 1
            ddos_fragments += flow_fragments

    benign_flows = total_flows - ddos_flows
    benign_fragments = total_fragments - ddos_fragments

    return (total_flows, ddos_flows, benign_flows), (total_fragments, ddos_fragments, benign_fragments)


def balance_dataset(flows, total_fragments=float('inf')):
    """
    Balance the dataset by sampling equal numbers of benign and DDoS fragments.

    Args:
        flows (List[Tuple]): List of flows
        total_fragments (int): Max number of fragments to keep

    Returns:
        Tuple:
            - balanced flow list
            - number of benign fragments
            - number of DDoS fragments
    """
    new_flow_list = []
    _, (_, ddos_fragments, benign_fragments) = count_flows(flows)

    if ddos_fragments == 0 or benign_fragments == 0:
        min_fragments = total_fragments
    else:
        min_fragments = min(total_fragments / 2, ddos_fragments, benign_fragments)

    random.shuffle(flows)
    new_benign_fragments = 0
    new_ddos_fragments = 0

    for flow in flows:
        fragment_count = len(flow[1]) - 1
        label = flow[1].get('label', 0)
        if label == 0 and new_benign_fragments < min_fragments:
            new_benign_fragments += fragment_count
            new_flow_list.append(flow)
        elif label > 0 and new_ddos_fragments < min_fragments:
            new_ddos_fragments += fragment_count
            new_flow_list.append(flow)

    return new_flow_list, new_benign_fragments, new_ddos_fragments


def dataset_to_list_of_fragments(dataset):
    """
    Convert flows into flat lists of fragments and labels.

    Args:
        dataset (List[Tuple[Tuple, Dict]]): Preprocessed flow dataset

    Returns:
        Tuple:
            - List of fragment arrays (X)
            - List of labels (y)
            - List of 5-tuples (keys)
    """
    keys = []
    X = []
    y = []

    for flow_id, flow_data in dataset:
        label = flow_data.get('label', 0)
        for key, fragment in flow_data.items():
            if key != 'label':
                X.append(fragment)
                y.append(label)
                keys.append(flow_id)

    return X, y, keys
