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

import pprint
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def report_results(Y_true, Y_pred, packets, model_name, data_source, prediction_time, writer):
    """
    Calculate metrics and log the prediction results to CSV and stdout.

    Args:
        Y_true (np.array or None): Ground truth labels
        Y_pred (np.array): Predicted labels (0 or 1)
        packets (int): Number of packets in the dataset
        model_name (str): Name of the model
        data_source (str): Source of the data (file name or interface)
        prediction_time (float): Time spent on prediction
        writer (csv.DictWriter): Writer for the output CSV file
    """
    ddos_rate = '{:04.3f}'.format(np.sum(Y_pred) / Y_pred.shape[0])

    if Y_true is not None and len(Y_true.shape) > 0:
        Y_true = Y_true.reshape((Y_true.shape[0], 1))
        accuracy = accuracy_score(Y_true, Y_pred)
        f1 = f1_score(Y_true, Y_pred)
        tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred, labels=[0, 1]).ravel()

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        tnr = tn / (tn + fp)
        fnr = fn / (fn + tp)

        row = {
            'Model': model_name,
            'Time': '{:04.3f}'.format(prediction_time),
            'Packets': packets,
            'Samples': Y_pred.shape[0],
            'DDOS%': ddos_rate,
            'Accuracy': '{:05.4f}'.format(accuracy),
            'F1Score': '{:05.4f}'.format(f1),
            'TPR': '{:05.4f}'.format(tpr),
            'FPR': '{:05.4f}'.format(fpr),
            'TNR': '{:05.4f}'.format(tnr),
            'FNR': '{:05.4f}'.format(fnr),
            'Source': data_source
        }
    else:
        row = {
            'Model': model_name,
            'Time': '{:04.3f}'.format(prediction_time),
            'Packets': packets,
            'Samples': Y_pred.shape[0],
            'DDOS%': ddos_rate,
            'Accuracy': "N/A",
            'F1Score': "N/A",
            'TPR': "N/A",
            'FPR': "N/A",
            'TNR': "N/A",
            'FNR': "N/A",
            'Source': data_source
        }

    pprint.pprint(row, sort_dicts=False)
    writer.writerow(row)
