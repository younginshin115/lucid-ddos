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

import glob
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)
from keras_tuner import RandomSearch
from model.builder import model_builder
from utils.path_utils import get_model_basename, get_model_path
from utils.logging_utils import save_evaluation_artifacts
from utils.callbacks import create_early_stopping_callback, create_model_checkpoint_callback, create_tensorboard_callback
from utils.constants import PATIENCE
from core.helpers import parse_training_filename, load_and_shuffle_dataset
from utils.class_weight_utils import get_class_weight

def evaluate_model(model, X_val, Y_val, label_mode="binary"):
    """
    Evaluate the trained model on the validation set.

    Args:
        model (keras.Model): Trained Keras model
        X_val (np.ndarray): Validation features
        Y_val (np.ndarray): Ground-truth validation labels (one-hot for multi)
        label_mode (str): "binary" or "multi"

    Returns:
        dict: Evaluation results including accuracy, F1/F1_weighted, precision, recall, confusion matrix
    """
    Y_pred_probs = model.predict(X_val)

    if label_mode == "multi":
        Y_pred = Y_pred_probs.argmax(axis=1)
        Y_true = Y_val.argmax(axis=1)
        average_type = "weighted"
    else:
        Y_pred = (Y_pred_probs > 0.5).astype(int).reshape((-1,))
        Y_true = Y_val.reshape((-1,))
        average_type = "binary"

    return {
        "accuracy": accuracy_score(Y_true, Y_pred),
        "f1": f1_score(Y_true, Y_pred, average=average_type),
        "precision": precision_score(Y_true, Y_pred, average=average_type, zero_division=0),
        "recall": recall_score(Y_true, Y_pred, average=average_type, zero_division=0),
        "confusion_matrix": confusion_matrix(Y_true, Y_pred).tolist(),  # convert to JSON serializable format
        "samples": len(Y_true)
    }

def run_training(args, output_folder):
    """
    Main training loop for one or multiple datasets in a given folder.

    Iterates over dataset subfolders, builds and trains models, evaluates them,
    saves the best model and logs metrics to CSV.

    Args:
        args (argparse.Namespace): Parsed CLI arguments
        output_folder (str): Folder path where models and logs will be saved
    """
    label_mode = args.label_mode # binary or multi
    
    subfolders = glob.glob(args.train[0] + "/*/")
    if len(subfolders) == 0: 
        subfolders = [args.train[0] + "/"]
    else:
        subfolders = sorted(subfolders)

    for dataset_folder in subfolders:
        dataset_folder = dataset_folder.replace("//", "/") # remove double slashes when needed
        print("\nCurrent dataset folder:", dataset_folder)

        (X_train, Y_train), (X_val, Y_val) = load_and_shuffle_dataset(
            dataset_folder + f"/*-{label_mode}-dataset-train.hdf5",
            dataset_folder + f"/*-{label_mode}-dataset-val.hdf5",
            label_mode=label_mode
        )

        # Calculate number of classes
        if label_mode == "multi":
            num_classes = Y_train.shape[1]
        else:
            num_classes = 1 # if binary, output neuron is 1

        # Extract hyperparameters from filename
        train_file = glob.glob(dataset_folder + "/*-train.hdf5")[0]
        time_window, max_flow_len, dataset_name = parse_training_filename(train_file)

        print ("\nCurrent dataset folder: ", dataset_folder)

        # Build model
        model_name = f"{dataset_name}-LUCID"
        model_basename = get_model_basename(time_window, max_flow_len, model_name, label_mode)
        best_model_path = get_model_path(output_folder, model_basename)
                
        # Create callbacks
        tensorboard_callback = create_tensorboard_callback(experiment_name=model_name)
        early_stopping_callback = create_early_stopping_callback(patience=PATIENCE)
        model_checkpoint_callback = create_model_checkpoint_callback(best_model_path)

        # Build callback list
        callbacks = [tensorboard_callback, early_stopping_callback, model_checkpoint_callback]

        # Initialize Keras Tuner (RandomSearch)
        tuner = RandomSearch(
            hypermodel=lambda hp: model_builder(
                hp, 
                input_shape=X_train.shape[1:], 
                label_mode=label_mode, 
                num_classes=num_classes
            ), # hp: hyperparameter
            objective='val_accuracy',  # Optimize for best validation accuracy
            max_trials=10,  # Try 10 different hyperparameter sets
            executions_per_trial=1,  # Train once per set
            directory=output_folder,  # Where to store tuning results
            project_name=model_basename  # Organize tuning projects
        )

        # Start hyperparameter search
        tuner.search(
            X_train, Y_train,
            epochs=args.epochs,
            validation_data=(X_val, Y_val),
            callbacks=callbacks
        )

        # Retrieve the best model after tuning
        best_model = tuner.get_best_models(num_models=1)[0]


        # Retrieve the best hyperparameters
        best_hyperparams = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Save the best model to disk
        best_model.save(best_model_path + ".h5")

        # Evaluate the best model
        metrics = evaluate_model(best_model, X_val, Y_val, label_mode=label_mode)

        # Save evaluation metrics
        save_evaluation_artifacts(
            base_path=best_model_path,
            model_name=model_name,
            metrics=metrics,
            used_hyperparams=best_hyperparams.values,
            val_file_path=glob.glob(dataset_folder + "/*-val.hdf5")[0],
            label_mode=label_mode,
            class_labels=list(range(num_classes)) if label_mode == "multi" else [0, 1]
        )


        print(f"[✓] Best parameters: {best_hyperparams.values}")
        print(f"[✓] Saved model to: {best_model_path}.h5")
        print(f"[✓] F1 Score on validation set: {metrics['f1']:.4f}")
