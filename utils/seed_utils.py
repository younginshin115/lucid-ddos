import os
import random as rn
import numpy as np
import tensorflow as tf
from utils.constants import SEED

def set_seed(seed=SEED):
    """
    Set random seeds across Python, NumPy, and TensorFlow for reproducibility.

    Args:
        seed (int): Seed value to use (defaults to module-level SEED)
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)
    tf.random.set_seed(seed)
