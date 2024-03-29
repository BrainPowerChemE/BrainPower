import unittest
import pandas as pd
import numpy as np
from statistics import stdev
import statsmodels
from statsmodels.sandbox.stats.multicomp import multipletests
import imblearn
import sys
from numpy import mean, absolute, sqrt
import scipy.stats
import sklearn
import sklearn.linear_model
import sklearn.neighbors
import sklearn.preprocessing
from xgboost import XGBClassifier
import sklearn.ensemble
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer
from sklearn.naive_bayes import GaussianNB
from itertools import combinations, cycle
import time

"""
Function: evaluate the performance of various machine learning classifiers with cross-validation
    
Input: balanced dvelopment Dataframe (data_dev.df), testing DataFrame (data_test.df) and selected
features list (selected_features.json)
    
Output: Dataframe (stats.df) store the evaluation results including the number of folds, scores,
average score, standard deviation, model used and method
"""

def apply_ml_model(dev, classifier, scoring_method='balanced_accuracy', cv=10, feature_list=None):
    """
    Finds the score for different ML classifiers
    Takes a dataframe with only target and feature columns
    dev : the development dataframe with a single categorical target column and float64 features
    classifier: ML classifier to test.
        options: "random_forest', "naive_bayes", "decision_tree"
    scoring_method: method of scoring the classifier
        run sklearn.metrics.get_scorer_names() to get a list of scoring methods
    target: target categorical column
    cv: folds to run in evaluation. takes integers or 'max': will run maximum number of folds = # of samples in categories
    """
    if feature_list is None:
        feature_list = selected_features
        
    # define predictor and response variables
    X = dev[feature_list]
    y = dev['group'].values.reshape(-1,1).ravel()
    
    # check that the data has equal number of categories in training data
    counts_list = list(dev['group'].value_counts())
    assert len(set(counts_list)) == 1, 'training data should contain equal quantities of categories. run bp_preprocessing.over_under() or other balancer'
    
    # choose model based on user input
    if classifier == "random_forest":
        model = sklearn.ensemble.RandomForestClassifier()
    elif classifier == "naive_bayes":
        model = sklearn.naive_bayes.GaussianNB()
    elif classifier == "decision_tree":
        model = sklearn.tree.DecisionTreeClassifier()
    elif classifier == "xgboost":
        model = XGBClassifier()
    else:
        raise ValueError("unknown classifier name provided")
    #define cv quantity:
    if type(cv) == int:
        pass
    elif cv == 'max':
        cv = counts_list[0]
    else:
        raise TypeError('Enter an integer or "max" as a string')
        
    scores = cross_val_score(model, X, y, scoring=scoring_method,cv=cv)
    mean_score = np.mean(abs(scores))
    std = np.std(scores)
    stats_list = [cv, scores, mean_score, std , model, scoring_method, feature_list[-2:]]
    stats_df = pd.DataFrame(data=[stats_list], columns=['folds','scores','abs_avg_score','std','model','scoring_method', 'features'])
    
    return stats_df
