import unittest
import pandas as pd
import numpy as np
from over_under import over_under
from sklearn.utils import check_random_state

class TestOverUnder(unittest.TestCase):

    def test_over_under(self):
        # Create a sample DataFrame for testing
        df_train = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'feature2': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'group': ['ClassA', 'ClassA', 'ClassB', 'ClassB', 'ClassC', 'ClassC']
        })
        
        # Apply the function
        result = over_under(df_train, cat_in_excess='ClassA', target='group', randomstate=42)
        
        # Check if the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check if the DataFrame has the correct columns
        self.assertEqual(result.columns.tolist(), ['group', 'feature1', 'feature2'])
        
        # Check if the excessive category is undersampled to equality
        self.assertEqual(result['group'].value_counts()['ClassA'], 2)
        
        # Check if the minority category is oversampled to equality
        self.assertEqual(result['group'].value_counts()['ClassB'], 2)
        self.assertEqual(result['group'].value_counts()['ClassC'], 2)
        
        # Check if the majority category remains the same
        self.assertEqual(result['group'].value_counts().index.tolist(), ['ClassA', 'ClassB', 'ClassC'])
