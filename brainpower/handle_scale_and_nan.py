import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn

    """
    Function: scale of numeric features and handle of missing values in the input
    
    Input: raw research_data.csv file with Patient’s ID number and biomakers
    
    Output: single formatted DataFrame (scaled_data_full.df)
    """

def handle_scale_and_nan(df):
    features = list(df.select_dtypes(include='float64'))
    cat = list(df.select_dtypes(include='object'))
    df = df.fillna(value=6)
    scaler = sklearn.preprocessing.StandardScaler().fit(df[features])
    df_cont = pd.DataFrame(data=scaler.transform(df[features]), columns=features)
    df_cat = pd.DataFrame(data=df[cat], columns=cat)
    df = pd.concat([df_cat,df_cont],axis=1)
    
    return df
