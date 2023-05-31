import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy
import statistics

import sklearn.linear_model
import sklearn.metrics
import sklearn.naive_bayes
import sklearn.preprocessing
import xgboost

import imblearn
import mrmr
from sklearn.model_selection import cross_val_score



def over_under(df_train,cat_in_excess='Healthy',target='group',randomstate=np.random.randint(0,4294967295)):
    """
    Function: perform oversampling and undersampling techniques to balance the distribution of a categorical variable in a dataset
    
    Input: formatted development DataFrame (data_dev.df)
    
    Output: balanced dvelopment Dataframe (data_dev.df)
    """
    # Drop the excessive category and oversample minority to the intermediate category
    df_train_no_excess = df_train[df_train.group != cat_in_excess]
    over_sampler = imblearn.over_sampling.RandomOverSampler(random_state=randomstate)
    X_train = df_train_no_excess.drop(columns=target)
    y_train = df_train_no_excess[target]
    X_train_over, y_train_over = over_sampler.fit_resample(X_train,y_train)
    df_train_over = pd.concat([y_train_over,X_train_over],axis=1)

    # Re-introduce the excessive category and undersample the majority to the minority
    df_train_excess = pd.concat([df_train_over,df_train[df_train[target] == cat_in_excess]])
    under_sampler = imblearn.under_sampling.RandomUnderSampler(random_state=randomstate)
    X_train = df_train_excess.drop(columns=target)
    y_train = df_train_excess[target]
    X_train_under, y_train_under = under_sampler.fit_resample(X_train,y_train)
    df_train_eq = pd.concat([y_train_under,X_train_under],axis=1)
    
    return df_train_eq
