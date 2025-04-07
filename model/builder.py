import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Conv2D
from tensorflow.keras.layers import Dropout, GlobalMaxPooling2D
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K


def Conv2DModel(model_name, input_shape, kernel_col, kernels=64, kernel_rows=3, regularization=None, dropout=None):
    """
    Build a simple Conv2D model for binary classification.
    """
    K.clear_session()

    inputs = Input(shape=input_shape, name="input")
    x = Conv2D(kernels, (kernel_rows, kernel_col), strides=(1, 1), kernel_regularizer=regularization, name='conv0')(inputs)
    if dropout is not None and isinstance(dropout, float):
        x = Dropout(dropout)(x)
    x = Activation('relu')(x)
    x = GlobalMaxPooling2D()(x)
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid', name='fc1')(x)

    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


def model_builder(**kwargs):
    """
    Filter valid hyperparameters, apply regularization if needed, 
    and return a compiled Conv2D model.
    """
    valid_keys = {
        "model_name",
        "input_shape",
        "kernel_col",
        "kernels",
        "kernel_rows",
        "learning_rate",
        "regularization",
        "dropout"
    }

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

    # Convert regularization string to actual keras regularizer
    reg = filtered_kwargs.get("regularization")
    if isinstance(reg, str):
        if reg == "l1":
            filtered_kwargs["regularization"] = regularizers.l1(0.01)
        elif reg == "l2":
            filtered_kwargs["regularization"] = regularizers.l2(0.01)

    return Conv2DModel(**filtered_kwargs)
