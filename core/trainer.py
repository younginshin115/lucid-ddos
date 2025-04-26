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

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score

from model.builder import model_builder
from utils.constants import PATIENCE, HYPERPARAM_GRID
from utils.path_utils import get_model_basename, get_model_path
from utils.logging_utils import save_metrics_to_csv, create_tensorboard_callback
from core.helpers import parse_training_filename, load_and_shuffle_dataset

def build_model(input_shape, kernel_col, model_name, args):
    """
    Build a KerasClassifier wrapper using the model_builder function.

    Args:
        input_shape (tuple): Input shape of the training data (excluding batch size)
        kernel_col (int): Number of kernel columns (used in model configuration)
        model_name (str): Identifier for the model (used in filename, metadata)
        args (argparse.Namespace): Parsed CLI arguments

    Returns:
        KerasClassifier: A compiled model wrapped for scikit-learn compatibility
    """
    return KerasClassifier(
        model=model_builder,
        model__model_name=model_name,
        model__input_shape=input_shape,
        model__kernel_col=kernel_col,
        epochs=args.epochs,
        verbose=1,
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
        compile=True
    )

def get_callbacks(model_path):
    """
    Create training callbacks for early stopping and model checkpoint saving.

    Args:
        model_path (str): Base path to save the best model (without extension)

    Returns:
        list: List of Keras callbacks (EarlyStopping, ModelCheckpoint)
    """
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)
    mc = ModelCheckpoint(
        filepath=model_path + ".h5",
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True
    )
    return [es, mc]

def train_model(model, X_train, Y_train, X_val, Y_val, callbacks, args):
    """
    Train the given model with or without cross-validation.

    Args:
        model (KerasClassifier): The model to train
        X_train (np.ndarray): Training input features
        Y_train (np.ndarray): Training labels
        X_val (np.ndarray): Validation input features
        Y_val (np.ndarray): Validation labels
        callbacks (list): List of Keras callbacks
        args (argparse.Namespace): Parsed CLI arguments

    Returns:
        tuple: (trained_model, used_hyperparameters)
    """
    if args.cross_validation >= 2:
        print(f"Cross-validation enabled with {args.cross_validation} folds.")
        cv = GridSearchCV(
            estimator=model,
            param_grid=HYPERPARAM_GRID,
            cv=args.cross_validation,
            refit=True,
            return_train_score=True
        )
        cv.fit(X_train, Y_train, callbacks=callbacks, validation_data=(X_val, Y_val))
        return cv.best_estimator_.model_, cv.best_params_
    else:
        print("Cross-validation disabled. Running single training cycle...")
        model.fit(X_train, Y_train, callbacks=callbacks, validation_data=(X_val, Y_val))
        return model.model_, {
            "optimizer": "adam",
            "loss": "binary_crossentropy",
            "metrics": ["accuracy"],
            "epochs": args.epochs
        }

def evaluate_model(model, X_val, Y_val):
    """
    Evaluate the trained model on the validation set.

    Args:
        model (keras.Model): Trained Keras model
        X_val (np.ndarray): Validation features
        Y_val (np.ndarray): Ground-truth validation labels

    Returns:
        dict: Evaluation results including accuracy, F1 score, and sample count
    """
    Y_pred = (model.predict(X_val) > 0.5)
    Y_true = Y_val.reshape((-1, 1))
    return {
        "f1": f1_score(Y_true, Y_pred),
        "accuracy": accuracy_score(Y_true, Y_pred),
        "samples": Y_pred.shape[0]
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
    subfolders = glob.glob(args.train[0] + "/*/")
    if len(subfolders) == 0: 
        subfolders = [args.train[0] + "/"]
    else:
        subfolders = sorted(subfolders)

    for dataset_folder in subfolders:
        dataset_folder = dataset_folder.replace("//", "/") # remove double slashes when needed
        print("\nCurrent dataset folder:", dataset_folder)

        (X_train, Y_train), (X_val, Y_val) = load_and_shuffle_dataset(dataset_folder + "/*-train.hdf5", dataset_folder + "/*-val.hdf5")

        # Extract hyperparameters from filename
        train_file = glob.glob(dataset_folder + "/*-train.hdf5")[0]
        time_window, max_flow_len, dataset_name = parse_training_filename(train_file)

        print ("\nCurrent dataset folder: ", dataset_folder)

        # Build model
        model_name = f"{dataset_name}-LUCID"
        model_basename = get_model_basename(time_window, max_flow_len, model_name)
        best_model_path = get_model_path(output_folder, model_basename)
        
        # Train model
        model = build_model(X_train.shape[1:], X_train.shape[2], model_name, args)

        # Create TensorBoard callback
        tensorboard_callback = create_tensorboard_callback(experiment_name=model_name)
        
        # Create other callbacks (ModelCheckpoint, EarlyStopping, etc.)
        callbacks = get_callbacks(best_model_path)
        
        # Append TensorBoard callback to the list
        callbacks.append(tensorboard_callback)
        print(callbacks)
        # Train model
        trained_model, used_hyperparams = train_model(model, X_train, Y_train, X_val, Y_val, callbacks, args)

        # Save the best trained model
        trained_model.save(best_model_path + ".h5")

        # Evaluate and log results
        metrics = evaluate_model(trained_model, X_val, Y_val)
        save_metrics_to_csv(
            best_model_path + ".csv",
            model_name,
            metrics,
            used_hyperparams,
            glob.glob(dataset_folder + "/*-val.hdf5")[0]
        )

        print(f"[✓] Best parameters: {used_hyperparams}")
        print(f"[✓] Saved model to: {best_model_path}.h5")
        print(f"[✓] F1 Score on validation set: {metrics['f1']:.4f}")
