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
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

import sklearn.ensemble
import time

def rule_based_classifier(df):
    mean = np.mean(df.iloc[:,1:], axis=0)
    std = np.std(df.iloc[:,1:], axis=0)
    zscores = (df.iloc[:,1:] - mean) / std
 
    cols_1 = ["TAU"]
    cols_2 = ["1433B", "1433G", "1433Z"]
    cols_3 = ["1433T", "GUAD"]
    cols_4 = ["AK1C1", "GDIA", "1433F", "SERC"]

    zscores_encoded = zscores
    for col in zscores.columns:
        if col in cols_1:
            zscores_encoded[col] = zscores[col]*4
        elif col in cols_2:
            zscores_encoded[col] = zscores[col]*3
        elif col in cols_3:
            zscores_encoded[col] = zscores[col]*2
        elif col in cols_4:
            zscores_encoded[col] = zscores[col]*1
        else:
            zscores_encoded[col] = zscores[col]*0

    zscores_encoded['Sum'] = zscores_encoded.sum(axis=1)
    zscores_encoded = pd.concat([df.iloc[:,:1], zscores_encoded], axis=1)
    return zscores_encoded

# dev = rule_based_classifier(dev)

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
        elif classifier == "xgboost":
            model = XGBClassifier()
        else: 
            print("wrong classifier named entered")
            
        model.fit(train_X, train_y) 
        fold_scores.append(
            sklearn.metrics.balanced_accuracy_score(val_y, model.predict(val_X)))
    return print(f"Leave-one-out cross validated balanced accuracy scores for {classifier} model (mean, std): ({np.mean(fold_scores)}, {np.std(fold_scores)})")
