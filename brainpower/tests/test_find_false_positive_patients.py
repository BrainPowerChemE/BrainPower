import unittest
import pandas as pd
import numpy as np
from brainpower import find_false_positive_patients

class TestFindFalsePositivePatients(unittest.TestCase):

    def test_find_false_positive_patients(self):
        # Create a sample metadata DataFrame
        metadata = pd.DataFrame({
            'Public Sample ID': ['ID1', 'ID2', 'ID3', 'ID4'],
            'Age': [30, 40, 50, 60],
            'Gender': ['M', 'F', 'F', 'M'],
            'Disease': ['Healthy', 'Diseased', 'Healthy', 'Diseased']
        })
        
        # Create a sample ml_results DataFrame
        ml_results = pd.DataFrame({
            'Predicted': ['Diseased', 'Diseased', 'Healthy', 'Healthy'],
            'Actual': ['Healthy', 'Diseased', 'Healthy', 'Diseased'],
            'patient_ID': ['ID1', 'ID2', 'ID3', 'ID4']
        })
        
        # Apply the function
        result = find_false_positive_patients(metadata, ml_results)
        
        # Check if the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check if the DataFrame has the correct columns
        expected_columns = ['Public Sample ID', 'Age', 'Gender', 'Disease']
        self.assertEqual(result.columns.tolist(), expected_columns)
        
        # Check if the false positives were identified correctly
        expected_ids = ['ID1']
        self.assertEqual(result['Public Sample ID'].tolist(), expected_ids)
        
        # Check if the corresponding information is correct
        expected_age = [30]
        expected_gender = ['M']
        expected_disease = ['Healthy']
        self.assertEqual(result['Age'].tolist(), expected_age)
        self.assertEqual(result['Gender'].tolist(), expected_gender)
        self.assertEqual(result['Disease'].tolist(), expected_disease)
