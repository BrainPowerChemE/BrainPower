import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statistics import stdev
import numpy as np
from scipy import stats
import statsmodels
from statsmodels.sandbox.stats.multicomp import multipletests
from matplotlib import pyplot

from sklearn.model_selection import cross_val_score

import scipy.stats
import sklearn.linear_model
import sklearn.neighbors
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

import sklearn.ensemble
import time

list_features = ['AK1C1',
 'TAU',
 '1433G',
 'SCUB1',
 'FMOD',
 'AMYP',
 'CRIS3',
 'MYDGF',
 'RARR2',
 'ATS8',
 'PGK1',
 '1433Z',
 'SV2A',
 'TRH',
 'GUAD',
 'HV69D',
 'CO7',
 'SERC']

def apply_ml_model(dev,classifier,scoring_method='balanced_accuracy',target='group', cv=10, feature_list=list_features):
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
    # define predictor and response variables
    X = dev[feature_list]
    y = dev[target]
    
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
        print("wrong classifier named entered")
    #define cv quantity:
    if type(cv) == int:
        pass
    elif cv == 'max':
        cv = counts_list[0]
    else:
        raise TypeError('Enter an integer or "max" as a string')
        
    scores = cross_val_score(model, X, y, scoring=scoring_method,cv=cv)
    mean_score = np.mean(absolute(scores))
    std = np.std(scores)
    stats_list = [cv, scores, mean_score, std , model, scoring_method, feature_list[-2:]]
    stats_df = pd.DataFrame(data=[stats_list], columns=['folds','scores','abs_avg_score','std','model','scoring_method', 'features'])
    
    return stats_df