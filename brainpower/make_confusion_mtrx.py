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

from brainpower import select_features

    """
    Function: generate a confusion matrix for a random forest classifier
    
    Input: formatted balanced development DataFrame(data_dev.df) and testing DataFrame (data_test.df)
    
    Output: confusion matrix visualization with balanced accuracy score and dataframe (ml_results.df) with predicted condition vs. actual condition
    """

def make_confusion_mtrx(dev, df_test, feature_list=None):    
    
    if feature_list is None:
        feature_list = selected_features
        
    test_X = df_test[feature_list]
    test_y = df_test['group']

    dev_X = dev[feature_list]
    dev_y = dev['group'] # 0th column is our target

    model = sklearn.ensemble.RandomForestClassifier()
    model.fit(dev_X, dev_y) 
    print('score=', sklearn.metrics.balanced_accuracy_score(test_y, model.predict(test_X)))
    ConfusionMatrixDisplay.from_estimator(model, test_X, test_y)
    plt.show()
    return pd.DataFrame({"Predicted" : model.predict(test_X), 'Actual': test_y, 'patient_ID': df_test['assay_ID']})                        
def main():
    PATH_DEV_DATA = sys.argv[1]
    PATH_TEST_DATA = sys.argv[2]
    data_dev = pd.read_csv(PATH_DEV_DATA)
    data_test = pd.read_csv(PATH_TEST_DATA)
    feature_list = select_features(data_dev, 18)
    make_confusion_mtrx(data_dev, data_test, feature_list=feature_list)

if __name__ == '__main__':
    main()
