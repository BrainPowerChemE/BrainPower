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

selected_features = ['NEUG', 'PPIA', 'SEM3G', 'EMIL3', 'AK1C1', 'LY86', '1433G', 'GLT18', 'KCC2D', 'SPRN', 'PGM1', 'ITM2B', 'TAU', 'AMPN', 'AB42/AB40', '1433Z', 'S38AA', 'DSG2']

classes_of_interest=['Healthy', 'PD_MCI_LBD', 'PD', 'AD_MCI']

def make_confusion_mtrx(dev, df_test, feature_list=None):
    if feature_list is None:
        feature_list = selected_features
        
    test_X = df_test[feature_list]
    test_y = df_test['group']

    dev_X = dev[feature_list]
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
