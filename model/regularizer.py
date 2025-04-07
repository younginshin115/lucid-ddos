from tensorflow.keras import regularizers

def get_regularizer(name):
    """
    Return a Keras regularizer object based on the given name.
    
    Args:
        name (str or None): One of 'l1', 'l2', or None.

    Returns:
        keras.regularizer or None
    """
    if name == "l1":
        return regularizers.l1(0.01)
    elif name == "l2":
        return regularizers.l2(0.01)
    else:
        return None
