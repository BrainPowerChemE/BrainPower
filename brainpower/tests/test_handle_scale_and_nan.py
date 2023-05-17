import unittest
import numpy as np
import pandas as pd
from handle_scale_and_nan import handle_scale_and_nan
from sklearn.preprocessing import StandardScaler


class TestHandleScaleAndNaN(unittest.TestCase):

    def test_handle_scale_and_nan(self):
        # Create a sample DataFrame for testing
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0],
            'C': [7.0, 8.0, 9.0],
            'D': ['X', 'Y', 'Z']
        })
        
        # Apply the function
        result = handle_scale_and_nan(df)
        
        # Check if the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check if the DataFrame has the correct columns
        self.assertEqual(result.columns.tolist(), ['D', 'A', 'B', 'C'])
        
        # Check if the categorical columns are unchanged
        self.assertTrue(result['D'].equals(df['D']))
        
        # Check if the continuous columns are standardized
        scaler = StandardScaler().fit(df[['A', 'B', 'C']])
        expected = pd.DataFrame(data=scaler.transform(df[['A', 'B', 'C']]), columns=['A', 'B', 'C'])
        self.assertTrue(result[['A', 'B', 'C']].equals(expected))
        
        # Check if NaN values are filled with 0.01
        self.assertTrue(result.isnull().sum().sum() == 0)
        
