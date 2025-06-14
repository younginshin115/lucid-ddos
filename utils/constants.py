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

SEED = 1
MAX_FLOW_LEN = 100 # number of packets
TIME_WINDOW = 10
TRAIN_SIZE = 0.90 # size of the training set wrt the total number of samples

PATIENCE = 10
DEFAULT_EPOCHS = 1000
VAL_HEADER = [
    'Model', 'Samples', 'Accuracy', 'F1Score', 'Precision', 'Recall',
    'Hyper-parameters', 'Validation Set', 'Label Mode'
]
PREDICT_HEADER = ['Model', 'Time', 'Packets', 'Samples', 'DDOS%', 'Accuracy', 'F1Score', 'TPR', 'FPR','TNR', 'FNR', 'Source']
PROTOCOLS = [
    'arp', 'data', 'dns', 'ftp', 'http', 'icmp',
    'ip', 'ssdp', 'ssl', 'telnet', 'tcp', 'udp'
]
POWERS_OF_TWO = np.array([2 ** i for i in range(len(PROTOCOLS))])