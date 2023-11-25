import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Charlotte.explor_visuals import Countplot_Visual,PiePlot


class TestCountplotVisual(unittest.TestCase):
    def test_single_variable_countplot(self):
        # Test case for single variable countplot with Pandas Series
        data_series = pd.Series(["A", "B", "A", "C", "B", "B", "A"], name="Categories")
        plot = Countplot_Visual(data_series)
        self.assertIsInstance(plot, plt.Figure)

    def test_single_variable_countplot_with_title(self):
        # Test case for single variable countplot with title
        data_array = np.array(["X", "Y", "X", "Y", "Y", "X", "X"])
        data_series = pd.Series(data_array, name="Categories")
        plot = Countplot_Visual(data_series, title="My Count Plot")
        self.assertIsInstance(plot, plt.Figure)

    def test_two_variable_countplot(self):
        # Test case for two-variable countplot with Pandas DataFrame
        data_df = pd.DataFrame(
            {
                "Category": ["A", "B", "A", "C", "B", "B", "A"],
                "Hue": ["X", "Y", "X", "Y", "Y", "X", "X"],
            }
        )
        plot = Countplot_Visual(data_df)
        self.assertIsInstance(plot, plt.Figure)

class TestPiePlotFunction(unittest.TestCase):

    def setUp(self):
        # Set up any necessary data for testing
        self.data = pd.Series([1, 2, 3, 1, 2, 1, 3, 3])

    def test_pie_plot(self):
        # Test with default parameters
        fig = PiePlot(self.data)
        self.assertIsInstance(fig, plt.Figure)

    def test_pie_plot_with_title(self):
        # Test with title parameter
        fig = PiePlot(self.data, title="Test Title")
        self.assertIsInstance(fig, plt.Figure)
        

if __name__ == "__main__":
    unittest.main()
