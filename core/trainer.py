import os
import glob
import csv

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score

from model.builder import model_builder
from data.data_loader import load_dataset
from utils.constants import SEED, PATIENCE, VAL_HEADER, HYPERPARAM_GRID
from utils.path_utils import get_model_basename, get_model_path

def build_model(input_shape, kernel_col, model_name, args):
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
    Y_pred = (model.predict(X_val) > 0.5)
    Y_true = Y_val.reshape((-1, 1))
    return {
        "f1": f1_score(Y_true, Y_pred),
        "accuracy": accuracy_score(Y_true, Y_pred),
        "samples": Y_pred.shape[0]
    }

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

        model = build_model(X_train.shape[1:], X_train.shape[2], model_name, args)
        callbacks = get_callbacks(best_model_path)
        trained_model, used_hyperparams = train_model(model, X_train, Y_train, X_val, Y_val, callbacks, args)

        trained_model.save(best_model_path + ".h5")

        metrics = evaluate_model(trained_model, X_val, Y_val)
        
        # 결과 저장
        with open(best_model_path + ".csv", 'w', newline='') as val_file:
            val_writer = csv.DictWriter(val_file, fieldnames=VAL_HEADER)
            val_writer.writeheader()

            row = {
                'Model': model_name,
                'Samples': metrics['samples'],
                'Accuracy': f"{metrics['accuracy']:05.4f}",
                'F1Score': f"{metrics['f1']:05.4f}",
                'Hyper-parameters': used_hyperparams,
                'Validation Set': glob.glob(dataset_folder + "/*-val.hdf5")[0]
            }

            val_writer.writerow(row)

        print(f"[✓] Best parameters: {used_hyperparams}")
        print(f"[✓] Saved model to: {best_model_path}.h5")
        print(f"[✓] F1 Score on validation set: {metrics['f1']:.4f}")
