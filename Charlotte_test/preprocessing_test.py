import unittest
import pandas as pd
import numpy as np
from Charlotte.preprocessing import fill_nan


class PreprocessingTest(unittest.TestCase):
    def test_calcMean(self):
        data = pd.DataFrame(
            {
                "A": [1, 2, np.nan, 4, 5],
                "B": ["a", "b", np.nan, "c", "a"],
                "C": [10.5, 20.3, np.nan, 15.8, 12.1],
            }
        )
        filled_data = fill_nan(data)
        self.assertFalse(filled_data.isnull().values.any())
