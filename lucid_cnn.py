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

#Sample commands
# Training: python3 lucid_cnn.py --train ./sample-dataset/  --epochs 100 -cv 5
# Testing: python3  lucid_cnn.py --predict ./sample-dataset/ --model ./sample-dataset/10t-10n-SYN2020-LUCID.h5

import tensorflow as tf
import os
import sys
from core.args import get_args
from core.trainer import run_training
from core.predictor import run_batch_prediction
from core.live_predictor import run_live_prediction

config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1)

import tensorflow.keras.backend as K

from utils.seed_utils import set_seed
set_seed()

K.set_image_data_format('channels_last')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)

from utils.path_utils import create_output_subfolder
OUTPUT_FOLDER = create_output_subfolder()

def main(argv):
    args = get_args()
    
    if os.path.isdir(OUTPUT_FOLDER) == False:
        os.mkdir(OUTPUT_FOLDER)

    if args.train is not None:
        run_training(args, OUTPUT_FOLDER)

    if args.predict is not None:
        run_batch_prediction(args, OUTPUT_FOLDER)

    if args.predict_live is not None:
        run_live_prediction(args, OUTPUT_FOLDER)

if __name__ == "__main__":
    main(sys.argv[1:])
