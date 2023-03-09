import pandas as pd
import numpy as np
import mrmr
import sklearn.metrics
import sklearn.linear_model
import seaborn as sns
import matplotlib.pyplot as plt 
import time

def mrmr_feature_selection(data_dev, split, 
    min_features, max_features, step_features, folds,
    tolerance=0.01,model=sklearn.linear_model.RidgeClassifier(),score=sklearn.metrics.balanced_accuracy_score):

    """
    data_dev: development data in a pandas dataframe
    split: split proportion of training and val data each time a fold is executed
    min_features: minimum number of features to return
    max_features: maximum number of features to return
    step_features: step between evaluation of features
    folds: number of repeat evaluations of one feature space size. The training and validation data will be resampled from the dev data each time. It is important that test data is not fed to this function. 
    tolerance: (Default=0.01) acceptable level of standard deviation between categories among the training and validation data. Percent categories are calculated first, then their stdev is calculated and compared to the stated tolerance


    """

    def mrmr_shorthand(X_train, y_train,X_val,y_val,K,model):
        feature_performance = []
        for k in K:
            reduced_features = mrmr.mrmr_classif(X_train,y_train,K=k)
            X_train_reduced = X_train[reduced_features]
            X_val_reduced = X_val[reduced_features]
            model.fit(X_train_reduced,y_train)
            feature_performance.append([k,score(y_val,model.predict(X_val_reduced)),reduced_features])
        return feature_performance
    t0 = time.time()
    folded_performances = []
    i=0
    while i < folds:
        # Split dev data
        data_train, data_val, randstate = split_cats_by_tolerance(data_dev,tolerance,silent=True,split=split)

        # Equalize the training data
        train_eq, randomstate = over_under(data_train,data_val)

        # Separate features from categories
        X_traineq = train_eq.drop(columns='group')
        y_traineq = train_eq['group']
        X_val = data_val.drop(columns='group')
        y_val = data_val['group']
        feature_performances = mrmr_shorthand(X_traineq,y_traineq,X_val,y_val,list(range(min_features,max_features+1,step_features)),model)
        folded_performances.append(feature_performances)
        i += 1

    folded_performances = pd.DataFrame(data=folded_performances)

    performance_mean = []
    for j in range(0,len(folded_performances.columns)):
        performance_values = []
        for i in range(0,len(folded_performances)):
            folded_list = folded_performances[j].iloc[i]
            featnum = folded_list[0]
            scores = folded_list[1]
            feats = folded_list[2]
            performance_values.append([featnum,scores,feats])

        scores = []
        for i in range(0,len(performance_values)):
            scores.append(performance_values[i][1])
        mean_score = np.mean(scores)

        features = []
        for i in range(0,len(performance_values)):
            features.append(performance_values[i][2])
        features = list(np.unique(features))

        std = np.std(scores)
        ci68 = scipy.stats.norm.interval(0.68, loc=mean_score, scale=std)
        yerr = float(np.diff(statstuple))/2
        
            
        performance_mean.append([featnum,scores,mean_score,std,ci68,yerr,features])
        number, scores, mean_score, std, ci68, yerr, features = zip(*performance_mean)

    performance_mean = pd.DataFrame(data=performance_mean,columns=['feature_num','ind_scores','avg_score','stdev','ci68','yerr','features'])
    plt.scatter(x=performance_mean['feature_num'],y=performance_mean['avg_score'],marker=".", s=1)
    plt.errorbar(x=performance_mean['feature_num'],y=performance_mean['avg_score'], c='black',elinewidth=1, yerr=performance_mean['yerr'], fmt="o")
    plt.show()
    t1 = time.time()
    total_time = t1-t0
    print('Time elapsed:',total_time)
    return performance_mean