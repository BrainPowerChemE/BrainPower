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

import scipy.stats
import sklearn.linear_model
import sklearn.neighbors

import sklearn.ensemble
import time

def apply_ml_model(dev, classifier):
    """
    finds the R2 score for different ML models
    """
    folds = sklearn.model_selection.LeaveOneOut().split(dev)

    fold_scores = []
    for train_indexes, val_indexes in folds: # the KFold splitter returns the indexes of the data, not the data itself
        train = dev.iloc[train_indexes]
        val = dev.iloc[val_indexes]

        scaler = sklearn.preprocessing.StandardScaler() # normalizing the numerical features
        train_X = scaler.fit_transform(train.iloc[:,1:]) 
        val_X = scaler.transform(val.iloc[:,1:])

        train_y = train.iloc[:,0] # 0th column is our target
        val_y = val.iloc[:,0]
        
        if classifier == "random_forest": 
            model = sklearn.ensemble.RandomForestClassifier()
        elif classifier == "naive_bayes":  
            model = sklearn.naive_bayes.GaussianNB()
        elif classifier == "decision_tree": 
            model = sklearn.tree.DecisionTreeClassifier()
        elif classifier == "rule_based": 
            # katherine's function
            print("missing rule-based code")
        else: 
            print("wrong classifier named entered")
            
        model.fit(train_X, train_y) 
        fold_scores.append(
            model.score(val_X, val_y))
    return print(f"Leave-one-out cross validated scores for {classifier} model (mean, std): ({np.mean(fold_scores)}, {np.std(fold_scores)})")
