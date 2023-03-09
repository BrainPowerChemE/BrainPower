import pandas as pd
import sklearn.preprocessing


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
        df = imputer.fit_transform(df[:, 1:])
    elif nan_decision == 'replace_ones': 
        df.fillna(value=1)
        
    return df
