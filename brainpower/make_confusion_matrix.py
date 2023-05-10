import imblearn
from itertools import combinations, cycle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import sklearn.ensemble
import sklearn.linear_model
from sklearn.metrics import RocCurveDisplay
import sklearn.neighbors
import sklearn.preprocessing
import statsmodels
import statistics
from numpy import absolute, mean, sqrt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve
from sklearn.model_selection import LeaveOneOut, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelBinarizer
from statsmodels.sandbox.stats.multicomp import multipletests
from xgboost import XGBClassifier

import sys

from apply_ml_model import apply_ml_model
from bp_preprocessing import handle_scale_and_nan, over_under

list_features = ['AK1C1','TAU', '1433G', 'SCUB1', 'FMOD', 'AMYP', 'CRIS3', 
                'MYDGF', 'RARR2', 'ATS8', 'PGK1', '1433Z', 'SV2A', 'TRH', 'GUAD', 
                'HV69D', 'CO7', 'SERC']

classes_of_interest=['Healthy', 'PD_MCI_LBD', 'PD', 'AD_MCI']

def pre_process_data(data_dev, data_test):
    data_dev = handle_scale_and_nan(data_dev)
    data_test = handle_scale_and_nan(data_test)
    data_dev = data_dev.drop(columns='assay_ID')
    data_test = data_test.drop(columns='assay_ID')
    data_dev = over_under(data_dev) #split groups equally
    return data_dev, data_test

def make_confusion_mtrx(dev, data_test, features=list_features):
    test_X = data_test[list_features]
    test_y = data_test['group']

    dev_X = dev[list_features]
    dev_y = dev.iloc[:,0] # 0th column is our target

    model = sklearn.ensemble.RandomForestClassifier()
    model.fit(dev_X, dev_y)
    print('score=', sklearn.metrics.balanced_accuracy_score(test_y, model.predict(test_X)))
    ConfusionMatrixDisplay.from_estimator(model, test_X, test_y)
    plt.show()

def main():
    PATH_DEV_DATA = sys.argv[1]
    PATH_TEST_DATA = sys.argv[2]
    data_dev = pd.read_csv(PATH_DEV_DATA)
    data_test = pd.read_csv(PATH_TEST_DATA)
    make_confusion_mtrx(data_dev, data_test, features=list_features)

if __name__ == '__main__':
    main()
