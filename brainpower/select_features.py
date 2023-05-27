    """
    Function: perform oversampling and undersampling techniques to balance the distribution of a categorical variable in a dataset
    
    Input: formatted development DataFrame (data_dev.df)
    
    Output: balanced dvelopment Dataframe (data_dev.df)
    """

def select_features(data_dev, n):
    X_dev = data_dev.iloc[:, 2:-1]
    y_dev = data_dev.iloc[:, 0]
    
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples = 100, n_features = 20, n_informative = 2, n_redundant = 2)
    X = pd.DataFrame(X_dev)
    y = pd.Series(y_dev)
    
    from mrmr import mrmr_classif
    feature_list = mrmr_classif(X=X, y=y, K=n)
    print(feature_list)
    
    import json
    aList = feature_list
    jsonStr = json.dumps(aList)
    
    jsonFile = open("mRMR_"+ str(n)+ "_features.json", "w")
    jsonFile.write(jsonStr)
    jsonFile.close()
    
    return feature_list
