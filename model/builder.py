from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Conv2D
from tensorflow.keras.layers import Dropout, GlobalMaxPooling2D
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from model.regularizer import get_regularizer

def Conv2DModel(model_name, input_shape, kernel_col, kernels=64, kernel_rows=3,
                regularization=None, dropout=None, num_classes=1):
    """
    Build a simple Conv2D model for binary or multi-class classification.

    Args:
        model_name (str): Name of the model.
        input_shape (tuple): Shape of the input data.
        kernel_col (int): Kernel width.
        kernels (int): Number of filters.
        kernel_rows (int): Kernel height.
        regularization (str or None): Type of regularization to apply ("l1", "l2", or None).
        dropout (float or None): Dropout rate.
        num_classes (int): Number of output classes (1 for binary classification).

    Returns:
        keras.Model: A compiled Keras model.
    """
    K.clear_session()

    regularizer = get_regularizer(regularization)

    inputs = Input(shape=input_shape, name="input")

    x = Conv2D(
        filters=kernels,
        kernel_size=(kernel_rows, kernel_col),
        strides=(1, 1),
        padding="same",
        kernel_regularizer=regularizer,
        name='conv0'
    )(inputs)

    if dropout is not None and isinstance(dropout, float) and dropout > 0.0:
        x = Dropout(dropout)(x)

    x = Activation('relu')(x)
    x = GlobalMaxPooling2D()(x)
    x = Flatten()(x)

    if num_classes == 1:
        outputs = Dense(1, activation='sigmoid', name='fc1')(x)
    else:
        outputs = Dense(num_classes, activation='softmax', name='fc1')(x)

    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    return model

def model_builder(hp, input_shape, label_mode = "binary", num_classes = 1):
    """
    Build and compile a Conv2D classification model with tunable hyperparameters.

    This function constructs a Conv2D model using the given input shape and
    hyperparameters, and compiles it with an appropriate loss function based on
    the label mode (binary or multi-class).

    Args:
        hp (kerastuner.HyperParameters): Hyperparameter search space.
        input_shape (tuple): Shape of the input data (excluding batch size).
        label_mode (str, optional): Type of classification. One of ['binary', 'multi'].
        num_classes (int, optional): Number of output classes. Set to 1 for binary classification.

    Returns:
        keras.Model: A compiled Keras model ready for training.
    """
    model = Conv2DModel(
        model_name=hp.Choice("model_name", ["default"]),
        input_shape=input_shape,
        kernel_col=hp.Int("kernel_col", 2, 8, step=2),  # 줄임!
        kernels=hp.Int("kernels", 32, 128, step=32),
        kernel_rows=hp.Choice("kernel_rows", [3, 5]),
        regularization=hp.Choice("regularization", ["l1", "l2", "none"]),
        dropout=hp.Float("dropout", 0.0, 0.5, step=0.1),
        num_classes=num_classes
    )

    learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="LOG")
    loss_function = "categorical_crossentropy" if label_mode == "multi" else "binary_crossentropy"

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss_function,
        metrics=["accuracy"]
    )
    return model
