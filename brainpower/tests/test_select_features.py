import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from mrmr import mrmr_classif
import json
from brainpower import select_features

class TestSelectFeatures(unittest.TestCase):

    def test_select_features(self):
        # Create a sample DataFrame for testing
        
        # Set a random seed for reproducibility
        np.random.seed(123)
        
        # Generate random data for the DataFrame
        n_samples = 100
        n_features = 4
        
        data_dev = pd.DataFrame({
        'Target': np.random.randint(0, 2, n_samples),
        'Feature1': np.random.randn(n_samples),
        'Feature2': np.random.randn(n_samples),
        'Feature3': np.random.randn(n_samples),
        'Feature4': np.random.randn(n_samples)
        })

        # Call the select_features function
        n = 2
        result = select_features(data_dev, n)

        # Assert that the result is a list with length equal to n
        self.assertEqual(len(result), n)

        # Assert that the JSON file is created and contains the selected features
        expected_features = data_dev.columns[2:2+n].tolist()
        with open(f"mRMR_{n}_features.json", "r") as json_file:
            json_data = json.load(json_file)
            self.assertListEqual(json_data, expected_features)
