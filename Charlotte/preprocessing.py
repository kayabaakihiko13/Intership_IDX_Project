import numpy as np
import pandas as pd
from scipy.stats import shapiro
from typing import Union


def fill_nan(data: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
    """
    ## Describe
    this function for fill NaN with method Shapiro (for Numerical)
    and Mode value (for Catogorical).

    Params:
        data (Union[pd.DataFrame, np.ndarray]): input data with 2D Shape

    Returns:
        Union[pd.DataFrame, np.ndarray]: data with 2D without NaN Value

    ## Example:
    >>> import numpy as np
    >>> data = np.random.rand(2,2)
    >>> data[:1,0] = np.NaN
    >>> result = fill_nan(data)
    >>> has_null = np.isnan(result).any()
    False

    """
    if isinstance(data, pd.DataFrame):
        filled_data = (
            data.copy()
        )  # Work on a copy to avoid modifying the original DataFrame
        for col in filled_data.columns:
            if filled_data[col].isnull().any():  # Check for NaN values in the column
                if not np.issubdtype(filled_data[col].dtype, np.number):
                    mode_val = filled_data[col].mode()[0]
                    filled_data[col].fillna(mode_val, inplace=True)
                else:
                    _, p = shapiro(filled_data[col])
                    if p > 0.05:
                        mean_val = filled_data[col].mean()
                        filled_data[col].fillna(mean_val, inplace=True)
                    else:
                        median_val = filled_data[col].median()
                        filled_data[col].fillna(median_val, inplace=True)
        return filled_data

    elif isinstance(data, np.ndarray):
        filled_data = (
            data.copy()
        )  # Work on a copy to avoid modifying the original array
        for i in range(filled_data.shape[1]):
            col = filled_data[:, i]
            if np.isnan(col).any():  # Check for NaN values in the column
                if not np.issubdtype(col.dtype, np.number):
                    # Implement your own mode calculation for NumPy array or handle categorical data differently
                    pass
                else:
                    _, p = shapiro(col)
                    if p > 0.05:
                        mean_val = np.nanmean(col)
                        col[np.isnan(col)] = mean_val
                    else:
                        median_val = np.nanmedian(col)
                        col[np.isnan(col)] = median_val
        return filled_data
