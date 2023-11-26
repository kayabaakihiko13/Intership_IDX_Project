import unittest
from Charlotte_test.preprocessing_test import PreprocessingTest
from Charlotte_test.explor_visual_test import TestCountplotVisual, TestPiePlotFunction
from Charlotte_test.feature_selection_test import TestFeatureSelectionMethods

if __name__ == "__main__":
    # Create instances of the test classes
    preprocessing_test_instance = unittest.TestLoader().loadTestsFromTestCase(
        PreprocessingTest
    )
    countplot_visual_instance = unittest.TestLoader().loadTestsFromTestCase(
        TestCountplotVisual
    )
    pie_plot_function_instance = unittest.TestLoader().loadTestsFromTestCase(
        TestPiePlotFunction
    )
    feature_selection_methods_instance = unittest.TestLoader().loadTestsFromTestCase(
        TestFeatureSelectionMethods
    )

    # Create a test suite combining all test instances
    test_suite = unittest.TestSuite(
        [
            preprocessing_test_instance,
            countplot_visual_instance,
            pie_plot_function_instance,
            feature_selection_methods_instance,
        ]
    )

    # Run the test suite
    unittest.TextTestRunner().run(test_suite)
