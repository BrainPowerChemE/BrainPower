import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from brainpower import make_confusion_mtrx
from brainpower import make_roc_curves

class TestMakeRocCurve(unittest.TestCase):

    def test_make_roc_curves(self):
        # Create a sample development dataset
        dev = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [5.0, 6.0, 7.0, 8.0],
            'group': ['A', 'B', 'A', 'B'],
            'assay_ID': ['ID1', 'ID2', 'ID3', 'ID4']
        })
        
        # Create a sample test dataset
        df_test = pd.DataFrame({
            'feature1': [1.5, 2.5, 3.5, 4.5],
            'feature2': [5.5, 6.5, 7.5, 8.5],
            'group': ['A', 'B', 'A', 'B'],
            'assay_ID': ['ID5', 'ID6', 'ID7', 'ID8']
        })
        
        
        data = {
            'Public Sample ID': ['ID1', 'ID2', 'ID3', 'ID4', 'ID5', 'ID6', 'ID7', 'ID8'],
            'Age': [66, 71, 75, 80, 68, 72, 78, 64],
            'Sex': ['Male', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Male']
        }

        df_metadata = pd.DataFrame(data)

        # Apply the function
        result = make_confusion_mtrx(dev, df_test, feature_list=['feature1', 'feature2'])
        roc_plot=make_roc_curves(data_dev=dev, data_test=df_test)
    
        # Check if the result is a DataFrame
        self.assertIsInstance(false_pos_df, pd.DataFrame)
