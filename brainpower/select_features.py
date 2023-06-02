import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from mrmr import mrmr_classif
import json

def select_features(data_dev, n):
    
    """
    Function: selecte the top N features using the mRMR (minimum redundancy maximum relevance) algorithm
    
    Input: balanced dvelopment Dataframe (data_dev.df)
    
    Output: a JSON file with selected N features (selected_features.json)
    """
    
    X_dev = data_dev.iloc[:, 2:-1]
    y_dev = data_dev.iloc[:, 0]
    
    
    X, y = make_classification(n_samples = 100, n_features = 20, n_informative = 2, n_redundant = 2)
    X = pd.DataFrame(X_dev)
    y = pd.Series(y_dev)
    
    
    feature_list = mrmr_classif(X=X, y=y, K=n)
    print(feature_list)
    
    
    aList = feature_list
    jsonStr = json.dumps(aList)
    
    jsonFile = open("mRMR_"+ str(n)+ "_features.json", "w")
    jsonFile.write(jsonStr)
    jsonFile.close()
    
    return feature_list
