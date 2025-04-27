import os
import tensorflow as tf

def run_tuner_and_fit_best_model(tuner, X_train, Y_train, X_val, Y_val, output_folder, epochs):
    """
    Run tuner search and fit the best model separately to a clean TensorBoard log directory.

    Args:
        tuner (keras_tuner.Tuner): Initialized tuner object.
        X_train (np.ndarray): Training inputs.
        Y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation inputs.
        Y_val (np.ndarray): Validation labels.
        output_folder (str): Base output folder path.
        epochs (int): Number of epochs to fit the best model.

    Returns:
        keras.Model: The best tuned model.
    """
    # 1. Run hyperparameter search
    tuner.search(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs
    )

    # 2. Retrieve the best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # 3. Set up a clean TensorBoard log dir
    best_tensorboard_logdir = os.path.join(output_folder, "tensorboard_best")
    os.makedirs(best_tensorboard_logdir, exist_ok=True)

    best_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=best_tensorboard_logdir)

    # 4. Fine-tune best model (optional: can re-train or fine-tune further)
    best_model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        callbacks=[best_tensorboard_callback]
    )

    return best_model
