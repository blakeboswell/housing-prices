import numpy as np


def rmsle(y, y0):
    """
    """
    return np.sqrt(np.mean(np.square(np.log1p(y) - np.log1p(y0))))