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
import os
import sys
import tensorflow as tf
import tensorflow.keras.backend as K

from core.args import get_args
from core.trainer import run_training
from core.predictor import run_batch_prediction
from core.live_predictor import run_live_prediction
from utils.seed_utils import set_seed
from utils.path_utils import create_output_subfolder

def setup_tensorflow_environment():
    """
    Configure TensorFlow backend settings (GPU memory, logging, etc.)
    """
    config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True  # dynamic GPU memory allocation
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    K.set_image_data_format('channels_last')
    set_seed()

def prepare_output_folder() -> str:
    """
    Create and return the experiment output folder
    """
    output_folder = create_output_subfolder()
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    return output_folder

def run_mode(args, output_folder):
    """
    Execute based on provided CLI arguments.
    """
    if args.train:
        run_training(args, output_folder)
    elif args.predict:
        run_batch_prediction(args, output_folder)
    elif args.predict_live:
        run_live_prediction(args, output_folder)
    else:
        print("No valid mode selected. Use --train, --predict, or --predict_live.")
        sys.exit(1)

def main(argv):
    setup_tensorflow_environment()
    args = get_args()
    output_folder = prepare_output_folder()
    run_mode(args, output_folder)

if __name__ == "__main__":
    main(sys.argv[1:])