import unittest
import pandas as pd
import numpy as np
from Charlotte.feature_selection import selection_by_variance,pearson_correlation_selection
class TestFeatureSelectionMethods(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        data = {
            'Feature1': [1, 2, 3, 4, 5],
            'Feature2': [5, 4, 3, 2, 1],
            'Feature3': [2, 2, 2, 2, 2],
            'Feature4': [1, 1, 1, 1, 1]
        }
        self.sample_data = pd.DataFrame(data)

    def test_select_features_by_pearson(self):
        # Test if the function returns a DataFrame
        selected_data = pearson_correlation_selection(self.sample_data)
        self.assertIsInstance(selected_data, pd.DataFrame)

        # Test if the correct features are selected based on correlation
        expected_columns = ['Feature1', 'Feature3', 'Feature4']
        self.assertCountEqual(selected_data.columns, expected_columns)

    def test_select_features_by_variance(self):
        # Test if the function returns a DataFrame
        selected_data = selection_by_variance(self.sample_data)
        self.assertIsInstance(selected_data, pd.DataFrame)

        # Test if the correct features are selected based on variance
        expected_columns = ['Feature1', 'Feature2']
        self.assertCountEqual(selected_data.columns, expected_columns)

if __name__ == '__main__':
    unittest.main()
