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
import numpy as np
import os
import csv
from utils.constants import SEED, PATIENCE, DEFAULT_EPOCHS, VAL_HEADER, PREDICT_HEADER, HYPERPARAM_GRID 
from utils.preprocessing import normalize_and_padding
from utils.minmax_utils import static_min_max
from data.data_loader import load_dataset, count_packets_in_dataset
from data.parser import parse_labels
from data.flow_utils import dataset_to_list_of_fragments
from data.live_process import process_live_traffic

config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1)

from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scikeras.wrappers import KerasClassifier

from sklearn.model_selection import GridSearchCV

import tensorflow.keras.backend as K

from utils.seed_utils import set_seed
set_seed()

K.set_image_data_format('channels_last')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
from model.builder import model_builder
from utils.logger import report_results

from utils.path_utils import create_output_subfolder, get_output_path
OUTPUT_FOLDER = create_output_subfolder()

def main(argv):
    help_string = 'Usage: python3 lucid_cnn.py --train <dataset_folder> -e <epocs>'

    parser = argparse.ArgumentParser(
        description='DDoS attacks detection with convolutional neural networks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-t', '--train', nargs='+', type=str,
                        help='Start the training process')

    parser.add_argument('-e', '--epochs', default=DEFAULT_EPOCHS, type=int,
                        help='Training iterations')

    parser.add_argument('-cv', '--cross_validation', default=0, type=int,
                        help='Number of folds for cross-validation (default 0)')

    parser.add_argument('-a', '--attack_net', default=None, type=str,
                        help='Subnet of the attacker (used to compute the detection accuracy)')

    parser.add_argument('-v', '--victim_net', default=None, type=str,
                        help='Subnet of the victim (used to compute the detection accuracy)')

    parser.add_argument('-p', '--predict', nargs='?', type=str,
                        help='Perform a prediction on pre-preprocessed data')

    parser.add_argument('-pl', '--predict_live', nargs='?', type=str,
                        help='Perform a prediction on live traffic')

    parser.add_argument('-i', '--iterations', default=1, type=int,
                        help='Predict iterations')

    parser.add_argument('-m', '--model', type=str,
                        help='File containing the model')

    parser.add_argument('-y', '--dataset_type', default=None, type=str,
                        help='Type of the dataset. Available options are: DOS2017, DOS2018, DOS2019, SYN2020')

    args = parser.parse_args()

    if os.path.isdir(OUTPUT_FOLDER) == False:
        os.mkdir(OUTPUT_FOLDER)

    if args.train is not None:
        subfolders = glob.glob(args.train[0] +"/*/")
        if len(subfolders) == 0: # for the case in which the is only one folder, and this folder is args.dataset_folder[0]
            subfolders = [args.train[0] + "/"]
        else:
            subfolders = sorted(subfolders)
        for full_path in subfolders:
            full_path = full_path.replace("//", "/")  # remove double slashes when needed
            folder = full_path.split("/")[-2]
            dataset_folder = full_path
            X_train, Y_train = load_dataset(dataset_folder + "/*" + '-train.hdf5')
            X_val, Y_val = load_dataset(dataset_folder + "/*" + '-val.hdf5')

            X_train, Y_train = shuffle(X_train, Y_train, random_state=SEED)
            X_val, Y_val = shuffle(X_val, Y_val, random_state=SEED)

            # get the time_window and the flow_len from the filename
            train_file = glob.glob(dataset_folder + "/*" + '-train.hdf5')[0]
            filename = train_file.split('/')[-1].strip()
            time_window = int(filename.split('-')[0].strip().replace('t', ''))
            max_flow_len = int(filename.split('-')[1].strip().replace('n', ''))
            dataset_name = filename.split('-')[2].strip()

            print ("\nCurrent dataset folder: ", dataset_folder)

            model_name = dataset_name + "-LUCID"
                        
            keras_classifier = KerasClassifier(
                model=model_builder,
                model__model_name=model_name,
                model__input_shape=X_train.shape[1:],
                model__kernel_col=X_train.shape[2],
                epochs=args.epochs,
                verbose=1,
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"],
                compile=True  # 🔥 이거 반드시 명시
            )

            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)
            
            model_basename = f"{time_window}t-{max_flow_len}n-{model_name}"
            best_model_filename = get_output_path(OUTPUT_FOLDER, model_basename)

            mc = ModelCheckpoint(
                filepath=best_model_filename + ".h5",
                monitor="val_accuracy",
                mode="max",
                verbose=1,
                save_best_only=True
            )

            if args.cross_validation >= 2:
                print(f"Cross-validation enabled with {args.cross_validation} folds.")

                rnd_search_cv = GridSearchCV(
                    estimator=keras_classifier,
                    param_grid=HYPERPARAM_GRID,
                    cv=args.cross_validation,
                    refit=True,
                    return_train_score=True
                )

                rnd_search_cv.fit(
                    X_train, Y_train,
                    callbacks=[es, mc],
                    validation_data=(X_val, Y_val)
                )

                best_model = rnd_search_cv.best_estimator_.model_
                best_model.save(best_model_filename + '.h5')

                Y_pred_val = (best_model.predict(X_val) > 0.5)
            else:
                print("Cross-validation disabled. Running single training cycle...")

                keras_classifier.fit(
                    X_train, Y_train,
                    callbacks=[es, mc],
                    validation_data=(X_val, Y_val)
                )

                keras_classifier.model_.save(best_model_filename + '.h5')
                Y_pred_val = (keras_classifier.predict(X_val) > 0.5)

            # With refit=True (default) GridSearchCV refits the model on the whole training set (no folds) with the best
            # hyper-parameters and makes the resulting model available as rnd_search_cv.best_estimator_.model
            if args.cross_validation >= 2:
                best_model = rnd_search_cv.best_estimator_.model_
            else:
                best_model = keras_classifier.model_


            # We overwrite the checkpoint models with the one trained on the whole training set (not only k-1 folds)
            best_model.save(best_model_filename + '.h5')

            # Alternatively, to save time, one could set refit=False and load the best model from the filesystem to test its performance
            #best_model = load_model(best_model_filename + '.h5')

            Y_pred_val = (best_model.predict(X_val) > 0.5)
            Y_true_val = Y_val.reshape((Y_val.shape[0], 1))
            f1_score_val = f1_score(Y_true_val, Y_pred_val)
            accuracy = accuracy_score(Y_true_val, Y_pred_val)

            # save best model performance on the validation set
            val_file = open(best_model_filename + '.csv', 'w', newline='')
            val_file.truncate(0)  # clean the file content (as we open the file in append mode)
            val_writer = csv.DictWriter(val_file, fieldnames=VAL_HEADER)
            val_writer.writeheader()
            val_file.flush()
            
            if args.cross_validation >= 2:
                hyperparams_used = rnd_search_cv.best_params_
            else:
                hyperparams_used = {
                    "optimizer": "adam",
                    "loss": "binary_crossentropy",
                    "metrics": ["accuracy"],
                    "epochs": args.epochs
                }
            row = {
                'Model': model_name,
                'Samples': Y_pred_val.shape[0],
                'Accuracy': '{:05.4f}'.format(accuracy),
                'F1Score': '{:05.4f}'.format(f1_score_val),
                'Hyper-parameters': hyperparams_used,
                'Validation Set': glob.glob(dataset_folder + "/*" + '-val.hdf5')[0]
            }
            val_writer.writerow(row)
            val_file.close()


            print("Best parameters: ", rnd_search_cv.best_params_) if args.cross_validation >= 2 else print("Default parameters used.")
            print("Best model path: ", best_model_filename)
            print("F1 Score of the best model on the validation set: ", f1_score_val)

    if args.predict is not None:
        predict_file = open(
            get_output_path(OUTPUT_FOLDER, f"predictions-{time.strftime('%Y%m%d-%H%M%S')}.csv"),
            'a',
            newline=''
        )
        predict_file.truncate(0)  # clean the file content (as we open the file in append mode)
        predict_writer = csv.DictWriter(predict_file, fieldnames=PREDICT_HEADER)
        predict_writer.writeheader()
        predict_file.flush()

        iterations = args.iterations

        dataset_filelist = glob.glob(args.predict + "/*test.hdf5")

        if args.model is not None:
            model_list = [args.model]
        else:
            model_list = glob.glob(args.predict + "/*.h5")

        for model_path in model_list:
            model_filename = model_path.split('/')[-1].strip()
            filename_prefix = model_filename.split('-')[0].strip() + '-' + model_filename.split('-')[1].strip() + '-'
            model_name_string = model_filename.split(filename_prefix)[1].strip().split('.')[0].strip()
            model = load_model(model_path)

            # warming up the model (necessary for the GPU)
            warm_up_file = dataset_filelist[0]
            filename = warm_up_file.split('/')[-1].strip()
            if filename_prefix in filename:
                X, Y = load_dataset(warm_up_file)
                Y_pred = np.squeeze(model.predict(X, batch_size=2048) > 0.5)

            for dataset_file in dataset_filelist:
                filename = dataset_file.split('/')[-1].strip()
                if filename_prefix in filename:
                    X, Y = load_dataset(dataset_file)
                    [packets] = count_packets_in_dataset([X])

                    Y_pred = None
                    Y_true = Y
                    avg_time = 0
                    for iteration in range(iterations):
                        pt0 = time.time()
                        Y_pred = np.squeeze(model.predict(X, batch_size=2048) > 0.5)
                        pt1 = time.time()
                        avg_time += pt1 - pt0

                    avg_time = avg_time / iterations

                    report_results(np.squeeze(Y_true), Y_pred, packets, model_name_string, filename, avg_time,predict_writer)
                    predict_file.flush()

        predict_file.close()

    if args.predict_live is not None:
        predict_file = open(
            get_output_path(OUTPUT_FOLDER, f"predictions-{time.strftime('%Y%m%d-%H%M%S')}.csv"),
            'a',
            newline=''
        )

        predict_file.truncate(0)  # clean the file content (as we open the file in append mode)
        predict_writer = csv.DictWriter(predict_file, fieldnames=PREDICT_HEADER)
        predict_writer.writeheader()
        predict_file.flush()

        if args.predict_live is None:
            print("Please specify a valid network interface or pcap file!")
            exit(-1)
        elif args.predict_live.endswith('.pcap'):
            pcap_file = args.predict_live
            cap = pyshark.FileCapture(pcap_file)
            data_source = pcap_file.split('/')[-1].strip()
        else:
            cap =  pyshark.LiveCapture(interface=args.predict_live)
            data_source = args.predict_live

        print ("Prediction on network traffic from: ", data_source)

        # load the labels, if available
        labels = parse_labels(args.dataset_type, args.attack_net, args.victim_net)

        # do not forget command sudo ./jetson_clocks.sh on the TX2 board before testing
        if args.model is not None and args.model.endswith('.h5'):
            model_path = args.model
        else:
            print ("No valid model specified!")
            exit(-1)

        model_filename = model_path.split('/')[-1].strip()
        filename_prefix = model_filename.split('n')[0] + 'n-'
        time_window = int(filename_prefix.split('t-')[0])
        max_flow_len = int(filename_prefix.split('t-')[1].split('n-')[0])
        model_name_string = model_filename.split(filename_prefix)[1].strip().split('.')[0].strip()
        model = load_model(args.model)

        mins, maxs = static_min_max(time_window)

        while (True):
            samples = process_live_traffic(cap, args.dataset_type, labels, max_flow_len, traffic_type="all", time_window=time_window)
            if len(samples) > 0:
                X,Y_true,keys = dataset_to_list_of_fragments(samples)
                X = np.array(normalize_and_padding(X, mins, maxs, max_flow_len))
                if labels is not None:
                    Y_true = np.array(Y_true)
                else:
                    Y_true = None

                X = np.expand_dims(X, axis=3)
                pt0 = time.time()
                Y_pred = np.squeeze(model.predict(X, batch_size=2048) > 0.5,axis=1)
                pt1 = time.time()
                prediction_time = pt1 - pt0

                [packets] = count_packets_in_dataset([X])
                report_results(np.squeeze(Y_true), Y_pred, packets, model_name_string, data_source, prediction_time,predict_writer)
                predict_file.flush()

            elif isinstance(cap, pyshark.FileCapture) == True:
                print("\nNo more packets in file ", data_source)
                break

        predict_file.close()

if __name__ == "__main__":
    main(sys.argv[1:])
