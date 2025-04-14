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
live_process.py

Functions to process live network traffic (or .pcap in streaming mode)
and transform them into preprocessed input samples for inference.
"""

import time
import pyshark
from collections import OrderedDict
from data.parser import parse_packet
from data.process_pcap import apply_labels  # 재사용
from utils.constants import TIME_WINDOW


def process_live_traffic(cap, dataset_type, in_labels, max_flow_len, traffic_type='all', time_window=TIME_WINDOW):
    """
    Process live or streaming pcap packets into flow structures usable for model inference.

    Args:
        cap (pyshark.LiveCapture or FileCapture): Live or static capture object
        dataset_type (str): Dataset type (used to fetch labels)
        in_labels (dict): Flow label mappings
        max_flow_len (int): Max packets per flow
        traffic_type (str): all / benign / ddos
        time_window (int): Flow grouping window (in seconds)

    Returns:
        List of labeled flows (same format as process_pcap output)
    """
    start_time = time.time()
    start_time_window = start_time
    window_end_time = start_time_window + time_window
    temp_dict = OrderedDict()
    labelled_flows = []

    if isinstance(cap, pyshark.LiveCapture):
        for pkt in cap.sniff_continuously():
            if time.time() >= window_end_time:
                break
            pf = parse_packet(pkt)
            from data.process_pcap import store_packet
            temp_dict = store_packet(pf, temp_dict, start_time_window, max_flow_len)
    elif isinstance(cap, pyshark.FileCapture):
        while time.time() < window_end_time:
            try:
                pkt = cap.next()
                pf = parse_packet(pkt)
                from data.process_pcap import store_packet
                temp_dict = store_packet(pf, temp_dict, start_time_window, max_flow_len)
            except:
                break

    apply_labels(temp_dict, labelled_flows, in_labels, traffic_type)
    return labelled_flows
