from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Conv2D
from tensorflow.keras.layers import Dropout, GlobalMaxPooling2D
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

def Conv2DModel(model_name, input_shape, kernel_col, kernels=64, kernel_rows=3, regularization=None, dropout=None):
    """
    Build a simple Conv2D model for binary classification.
    """
    K.clear_session()

    # Handle regularization
    if regularization == "l1":
        regularizer = l1(1e-4)
    elif regularization == "l2":
        regularizer = l2(1e-4)
    else:
        regularizer = None

    inputs = Input(shape=input_shape, name="input")

    x = Conv2D(
        filters=kernels,
        kernel_size=(kernel_rows, kernel_col),
        strides=(1, 1),
        padding="same",  # <-- 이거 추가해서 사이즈 맞춤
        kernel_regularizer=regularizer,
        name='conv0'
    )(inputs)

    if dropout is not None and isinstance(dropout, float) and dropout > 0.0:
        x = Dropout(dropout)(x)

    x = Activation('relu')(x)
    x = GlobalMaxPooling2D()(x)
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid', name='fc1')(x)

    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    return model

def model_builder(hp, input_shape):
    """
    Build a Conv2D model with tunable hyperparameters using KerasTuner.

    Args:
        hp (kerastuner.HyperParameters): Hyperparameter search space.
        input_shape (tuple): Shape of the input data.

    Returns:
        keras.Model: Compiled Conv2D model.
    """
    model = Conv2DModel(
        model_name=hp.Choice("model_name", ["default"]),
        input_shape=input_shape,
        kernel_col=hp.Int("kernel_col", 2, 8, step=2),  # 줄임!
        kernels=hp.Int("kernels", 32, 128, step=32),
        kernel_rows=hp.Choice("kernel_rows", [3, 5]),
        regularization=hp.Choice("regularization", ["l1", "l2", "none"]),
        dropout=hp.Float("dropout", 0.0, 0.5, step=0.1)
    )

    learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="LOG")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model
