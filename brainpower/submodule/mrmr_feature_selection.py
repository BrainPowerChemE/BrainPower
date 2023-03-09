import pandas as pd
import numpy as np
import mrmr
import sklearn.metrics
import sklearn.linear_model
import seaborn as sns


def mrmr_feature_selection(X_train, y_train,X_val,y_val,K,model):

    feature_performance = []
    for k in K:
        reduced_features = mrmr.mrmr_classif(X_train,y_train,K=k)
        X_train_reduced = X_train[reduced_features]
        X_val_reduced = X_val[reduced_features]
        model.fit(X_train_reduced,y_train)
        feature_performance.append([k,sklearn.metrics.balanced_accuracy_score(y_val,model.predict(X_val_reduced)),reduced_features])
    #return feature_performance
    folded_performances = []
    i=0
    while i < 5:
        # Split dev data
        data_train, data_val, randstate = split_cats_by_tolerance(data_dev,0.01,silent=True)

        # Equalize the training data
        train_eq, randomstate = over_under(data_train,data_val)

        # Separate features from categories
        X_traineq = train_eq.drop(columns='group')
        y_traineq = train_eq['group']
        X_val = data_val.drop(columns='group')
        y_val = data_val['group']
        feature_performances = mrmr_feature_selection(X_traineq,y_traineq,X_val,y_val,range(1,100),sklearn.linear_model.RidgeClassifier())
        folded_performances.append(feature_performances)
        i += 1

    folded_performances = pd.DataFrame(data=folded_performances)

    performance_mean = []
    for j in range(0,len(folded_performances.columns)):
        listy = []
        for i in range(0,len(folded_performances)):
            folded_list = folded_performances[j].iloc[i]
            a = folded_list[0]
            b = folded_list[1]
            c = folded_list[2]
            listy.append([a,b,c])

        meanie = []
        for i in range(0,len(listy)):
            meanie.append(listy[i][1])
        meanie = np.mean(meanie)

        featies = []
        for i in range(0,len(listy)):
            featies.append(listy[i][2])
        featies = list(np.unique(featies))

        performance_mean.append([a,meanie,featies])
        number, score, names = zip(*performance_mean)
        sns.scatterplot(x=number,y=score)

        return performance_mean