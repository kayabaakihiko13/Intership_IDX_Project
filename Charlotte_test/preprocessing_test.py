import unittest
import numpy as np
from Charlotte.preprocessing import calculate_mean


class PreprocessingTest(unittest.TestCase):
    def test_calcMean(self):
        input_data = np.array([1, 2, 3, 4])
        result = calculate_mean(input_data)
        expeted = 2.5
        self.assertEqual(result, expeted)
