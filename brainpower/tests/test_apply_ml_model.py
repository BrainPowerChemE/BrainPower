import unittest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from unittest import mock
from brainpower import apply_ml_model

class TestApplyMLModel(unittest.TestCase):
   def test_apply_ml_model(self):
        # Create a sample development DataFrame
        data_dev = pd.DataFrame({
            'group': ['A','A', 'A', 'A', 'A','A','A','A','A','A','A','A'],
            'Feature1': [1.2, 2.1, 3.3, 4.7, 6.4, 7.9, 8.2, 9.6, 13.2, 14.9, 12.0, 13.4],
            'Feature2': [6.4, 7.9, 8.2, 9.6, 4.7, 6.4, 7.9, 8.2, 9.6, 13.2, 45, 34],
            'Feature3': [11.3, 12.7, 13.2, 14.9, 8.2, 9.6, 4.7, 6.4, 7.9, 8.2, 23, 55],
            'Feature4': [16.6, 17.4, 18.8, 19.3, 8.2, 9.6, 4.7, 6.4, 7.9, 8.2, 34, 52]
        })
        selected_features = ['Feature1', 'Feature2']
                
        # Call the apply_ml_model function with the inputs
        result = apply_ml_model.apply_ml_model(data_dev, classifier='random_forest',scoring_method='balanced_accuracy', cv=10, feature_list=selected_features)

        # Assert that the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Assert that there is a number under the abs_avg_score column
        self.assertTrue("abs_avg_score" in result.columns)
        abs_avg_score = result["abs_avg_score"].values[0]
        self.assertTrue(isinstance(abs_avg_score, (float)))
