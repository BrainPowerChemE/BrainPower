import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

import sklearn.linear_model
import sklearn.metrics
import sklearn.naive_bayes
import sklearn.preprocessing

import imblearn
import mrmr


def handle_scale_and_nan(df, nan_decision='drop'):
    features = list(df.select_dtypes(include='float64'))
    cat = list(df.select_dtypes(include='object'))
    scaler = sklearn.preprocessing.StandardScaler().fit(df[features])
    df_cont = pd.DataFrame(data=scaler.transform(df[features]), columns=features)
    df_cat = pd.DataFrame(data=df[cat], columns=cat)
    
    df = pd.concat([df_cat,df_cont],axis=1)
    
    if nan_decision == 'mean':
        for feature in features:
            df[feature].fillna((df[feature].mean()), inplace=True)
    elif nan_decision == 'drop':
            df = df.dropna(axis=1)
    elif nan_decision == 'impute':
        imputer = missingpy.MissForest() #must be in shape of n_samples by n_features
        df = imputer.fit_transform(df[:, 1:]) # impute NaNs with existing numerical vals
        df.insert(0, "group", df[:, 0], allow_duplicates=True) # reinsert the nominal vals
    elif nan_decision == 'replace_ones': 
        df.fillna(value=1)
        
    return df

def split_cats_by_tolerance(frame,tolerance,silent=False,randomstate=98281,split=0.15,step=1,categories=['Healthy','AD_MCI','PD','PD_MCI_LBD']):
    tolerable_list =[]
    if randomstate == None:
        randomstate=np.random.randint(0,2**31)
    elif type(randomstate) == int:
        pass
    while sum(tolerable_list) != 4:
        df_dev, df_test = sklearn.model_selection.train_test_split(frame,test_size=split,random_state=randomstate)
        
        dev_dict = dict(df_dev['group'].value_counts())
        test_dict = dict(df_test['group'].value_counts())
        
        tolerable_list = []
        stats_dict ={}
        for i in range(0,len(categories)):
            try:
                percents = [(dev_dict[categories[i]]/len(df_dev)),(test_dict[categories[i]]/len(df_test))]
            except:
                break
            standdev = np.std(percents)
            if standdev <= tolerance:
                tolerable_list.append(1)
                stats_dict[str(categories[i])] = [[*percents],standdev]
            else:
                tolerable_list.append(0)
                
        randomstate += step

    if sum(tolerable_list) == 4:
        if silent == False:
            print(dev_dict)
            print(test_dict)
            print('Randstate:',randomstate-1)
            for i in range(0,len(categories)):            
                print('\nPercent',categories[i],'in dev, test:',stats_dict[categories[i]][0],
                      '\nStandard deviation of these values:',stats_dict[categories[i]][1],'\n')
        elif silent == True:
            pass
            
    return df_dev, df_test


def over_under(df_train,cat_in_excess='Healthy',target='group',randomstate=np.random.randint(0,4294967295)):
    """
    Takes dataframe(s) with only the target value and float64 features
    This function is to balance the samples in an imbalanced training dataset that has one category in excess, with additional categories more near each other
    The categories below the category in excess will be oversampled to equality, then the category in excess will be undersampled to equality
    ---Parameters---
    df_train: the training dataframe
    cat_in_excess: the category which is present in excess, far above the other categories
    target: target column in the dataframe
    randomstate: if chosen, this will the random state for the sampling. Default: None, numpy random integer method between 0 and 4294967295, the range of the sampling module used
    randomstate_sampler: the number of loops to run to compare random states starting from 
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

    print('randomstate')
    
    return df_train_eq


def mrmr_shorthand(X_train, y_train,X_val,y_val,K,model,score):
    feature_performance = []

    for k in K:
        reduced_features = mrmr.mrmr_classif(X_train,y_train,K=k)
        X_train_reduced = X_train[reduced_features]
        X_val_reduced = X_val[reduced_features]
        model.fit(X_train_reduced,y_train)
        feature_performance.append([k,score(y_val,model.predict(X_val_reduced)),reduced_features])
    return feature_performance

def mrmr_feature_selection(data_dev, split, 
    min_features, max_features, step_features, folds, tolerance=0.01, featureselector=mrmr.mrmr_classif, model=sklearn.linear_model.RidgeClassifier(), score=sklearn.metrics.balanced_accuracy_score):

    """
    data_dev: development data in a pandas dataframe
    split: split proportion of training and val data each time a fold is executed
    min_features: minimum number of features to return
    max_features: maximum number of features to return
    step_features: step between evaluation of features
    folds: number of repeat evaluations of one feature space size. The training and validation data will be resampled from the dev data each time. It is important that test data is not fed to this function. 
    tolerance: (Default=0.01) acceptable level of standard deviation between categories among the training and validation data. Percent categories are calculated first, then their stdev is calculated and compared to the stated tolerance

    """
    t0 = time.time()
    folded_performances = []
    i=0
    while i < folds:
        # Split dev data
        data_train, data_val = split_cats_by_tolerance(data_dev,tolerance,silent=True,split=split)

        # Equalize the training data
        train_eq = over_under(data_train,data_val,silent=True)

        # Separate features from categories
        X_traineq = train_eq.drop(columns='group')
        y_traineq = train_eq['group']
        X_val = data_val.drop(columns='group')
        y_val = data_val['group']
        feature_performances = mrmr_shorthand(X_traineq,y_traineq,X_val,y_val,list(range(min_features,max_features+1,step_features)),model,score)
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
        #features = list(np.unique(features))
        
        std = np.std(scores)
        ci68 = scipy.stats.norm.interval(0.68, loc=mean_score, scale=std)
        yerr = float(np.diff(ci68))/2
        
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