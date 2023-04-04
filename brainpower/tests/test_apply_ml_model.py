from apply_ml_model import apply_ml_model
from bp_preprocessing import over_under

import unittest
import pandas as pd



class TestApplyMLModel(unittest.TestCase):

    def setUp(self):
        self.dev = pd.read_csv('https://github.com/xsam0510/Class_ex/blob/main/Data_for_ML/dev.csv?raw=True')
        self.target = 'group'
        self.feature_list = ['AK1C1', 'TAU', '1433G', 'SCUB1', 'FMOD', 'AMYP', 'CRIS3', 'MYDGF', 'RARR2', 'ATS8', 'PGK1', '1433Z', 'SV2A', 'TRH', 'GUAD', 'HV69D', 'CO7', 'SERC']
        self.classifier = 'random_forest'
        self.scoring_method = 'balanced_accuracy'
        self.cv = 10
        self.dev = over_under(self.dev, self.target)
        
    def test_apply_ml_model(self):
        result = apply_ml_model(self.dev, self.classifier, self.scoring_method, self.target, self.cv, self.feature_list)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], 7)
        self.assertEqual(result.columns.tolist(), ['folds', 'scores', 'abs_avg_score', 'std', 'model', 'scoring_method', 'features'])
