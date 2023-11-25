import pandas as pd
import numpy as np

def pearson_correlation_selection(data: pd.DataFrame,
                                  threshold: float = 0.8)->pd.DataFrame:
    """
    Seleksi fitur berdasarkan korelasi Pearson antar fitur.

    Args:
    - data: pd.DataFrame, dataset yang akan digunakan
    - threshold: float, ambang batas korelasi yang ditetapkan untuk seleksi fitur

    Returns:
    - selected_features: list, fitur-fitur yang dipilih
    """
    # Matriks korelasi
    corr_matrix = data.corr().abs()

    # Matriks segitiga atas dari korelasi absolut
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Fitur-fitur yang ingin di-drop
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

    selected_features = [col for col in data.columns if col not in to_drop]

    return data[selected_features]


def selection_by_variance(data:pd.DataFrame,
                          threshold:float=0.8)->pd.DataFrame:
    
    
    variances = data.var()
    selected_features = data.columns[variances > threshold]
    return data[selected_features]

