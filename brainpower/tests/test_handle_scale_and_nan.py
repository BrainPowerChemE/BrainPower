import unittest
import pandas as pd
from sklearn.preprocessing import StandardScaler
from brainpower import handle_scale_and_nan

class TestHandleScaleAndNan(unittest.TestCase):

    def test_handle_scale_and_nan(self):
        # Create a sample DataFrame for testing
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0, 4.0, None],
            'B': [5.0, 6.0, 7.0, None, 9.0],
            'C': [10.0, None, 12.0, 13.0, 14.0],
            'D': ['a', 'b', 'c', 'd', 'e']
        })
        
        # Apply the function
        result = handle_scale_and_nan(df)
        
        # Check if the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check if the DataFrame has the correct columns
        self.assertEqual(result.columns.tolist(), ['D', 'A', 'B', 'C'])
        
        # Check if the missing values are filled with the specified value
        self.assertEqual(result['A'].isnull().sum(), 0)
        self.assertEqual(result['B'].isnull().sum(), 0)
        self.assertEqual(result['C'].isnull().sum(), 0)
        
        # Check if the scaling is applied correctly
        scaler = StandardScaler()
        expected = pd.DataFrame(data=scaler.fit_transform(df.fillna(value=6)[['A', 'B', 'C']]), columns=['A', 'B', 'C'])
        pd.testing.assert_frame_equal(result[['A', 'B', 'C']], expected)
