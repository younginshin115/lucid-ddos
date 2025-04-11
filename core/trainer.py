import os
import glob
import csv
import time
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score
from tensorflow.keras.models import load_model

from model.builder import model_builder
from data.data_loader import load_dataset
from utils.constants import SEED, PATIENCE, VAL_HEADER, HYPERPARAM_GRID
from utils.path_utils import get_model_basename, get_model_path
from utils.logger import report_results  # 또는 utils.eval_logger 이름에 따라
from utils.seed_utils import set_seed

set_seed()

def run_training(args, output_folder):
    subfolders = glob.glob(args.train[0] + "/*/")
    if len(subfolders) == 0: # for the case in which the is only one folder, and this folder is args.dataset_folder[0]
        subfolders = [args.train[0] + "/"]
    else:
        subfolders = sorted(subfolders)

    for dataset_folder in subfolders:
        dataset_folder = dataset_folder.replace("//", "/") # remove double slashes when needed
        print("\nCurrent dataset folder:", dataset_folder)

        X_train, Y_train = load_dataset(dataset_folder + "/*-train.hdf5")
        X_val, Y_val = load_dataset(dataset_folder + "/*-val.hdf5")

        X_train, Y_train = shuffle(X_train, Y_train, random_state=SEED)
        X_val, Y_val = shuffle(X_val, Y_val, random_state=SEED)

        # 파라미터 추출
        train_file = glob.glob(dataset_folder + "/*-train.hdf5")[0]
        filename = os.path.basename(train_file)
        time_window = int(filename.split('-')[0].replace('t', ''))
        max_flow_len = int(filename.split('-')[1].replace('n', ''))
        dataset_name = filename.split('-')[2]

        print ("\nCurrent dataset folder: ", dataset_folder)

        model_name = f"{dataset_name}-LUCID"
        model_basename = get_model_basename(time_window, max_flow_len, model_name)
        best_model_path = get_model_path(output_folder, model_basename)

        # 모델 구성
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
            compile=True
        )

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE)
        mc = ModelCheckpoint(
            filepath=best_model_path + ".h5",
            monitor="val_accuracy",
            mode="max",
            verbose=1,
            save_best_only=True
        )

        # 학습
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
        else:
            print("Cross-validation disabled. Running single training cycle...")
            keras_classifier.fit(
                X_train, Y_train,
                callbacks=[es, mc],
                validation_data=(X_val, Y_val)
            )
            best_model = keras_classifier.model_

        best_model.save(best_model_path + ".h5")

        Y_pred_val = (best_model.predict(X_val) > 0.5)
        Y_true_val = Y_val.reshape((-1, 1))
        f1_score_val = f1_score(Y_true_val, Y_pred_val)
        accuracy = accuracy_score(Y_true_val, Y_pred_val)

        # 결과 저장
        with open(best_model_path + ".csv", 'w', newline='') as val_file:
            val_writer = csv.DictWriter(val_file, fieldnames=VAL_HEADER)
            val_writer.writeheader()

            hyperparams_used = rnd_search_cv.best_params_ if args.cross_validation >= 2 else {
                "optimizer": "adam",
                "loss": "binary_crossentropy",
                "metrics": ["accuracy"],
                "epochs": args.epochs
            }

            row = {
                'Model': model_name,
                'Samples': Y_pred_val.shape[0],
                'Accuracy': f"{accuracy:05.4f}",
                'F1Score': f"{f1_score_val:05.4f}",
                'Hyper-parameters': hyperparams_used,
                'Validation Set': glob.glob(dataset_folder + "/*-val.hdf5")[0]
            }
            val_writer.writerow(row)

        print("Best parameters:", hyperparams_used)
        print("Best model path:", best_model_path)
        print("F1 Score on validation set:", f1_score_val)
