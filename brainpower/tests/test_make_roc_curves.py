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

import os 

class TestMakeRocCurve(unittest.TestCase):

    def test_make_roc_curves(self):
        # Create a sample development dataset
        dev = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [5.0, 6.0, 7.0, 8.0],
            'group': ['Healthy', 'PD_MCI_LBD', 'AD-MCI', 'PD'],
            'assay_ID': ['ID1', 'ID2', 'ID3', 'ID4']
        })

        # Create a sample test dataset with matching 'assay_ID' values
        df_test = pd.DataFrame({
            'feature1': [1.5, 2.5, 3.5, 4.5],
            'feature2': [5.5, 6.5, 7.5, 8.5],
            'group': ['Healthy', 'PD_MCI_LBD', 'PD', 'AD-MCI'],
            'assay_ID': ['ID1', 'ID2', 'ID3', 'ID4']  # Match 'assay_ID' values with data_dev
        })
       
        # Apply the function
        roc_plot=make_roc_curves(data_dev=dev, data_test=df_test, feature_list=['feature1', 'feature2'])
    
        # Check if the result is a file
        assert os.path.exists('roc_curves.png'), f"Failed to save {'roc_curves.png'}"
