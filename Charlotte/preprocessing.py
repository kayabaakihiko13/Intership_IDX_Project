import numpy as np

def calculate_mean(data:np.array)->float:
    """
    calculate mean
    Args:
        data (np.array): input data with array numpy type

    Returns:
        float : result
    """
    return np.sum(data) / data.shape[0]