import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def get_class_weight(y, label_mode="binary"):
    """
    Compute class weights for imbalanced classification.

    Args:
        y (np.ndarray): Labels, either 1D (binary) or 2D one-hot (multi-class).
        label_mode (str): 'binary' or 'multi'

    Returns:
        dict: Mapping of class index to weight, e.g., {0: 0.5, 1: 2.0}
    """
    if label_mode == "multi" and len(y.shape) == 2:
        y = np.argmax(y, axis=1)  # Convert one-hot to class indices
    elif label_mode == "binary":
        y = y.flatten()

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y
    )
    return dict(enumerate(class_weights))
