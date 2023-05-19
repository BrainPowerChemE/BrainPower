import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn

def handle_scale_and_nan(df):
    features = list(df.select_dtypes(include='float64'))
    cat = list(df.select_dtypes(include='object'))
    scaler = sklearn.preprocessing.StandardScaler().fit(df[features])
    df_cont = pd.DataFrame(data=scaler.transform(df[features]), columns=features)
    df_cat = pd.DataFrame(data=df[cat], columns=cat)
    df = pd.concat([df_cat,df_cont],axis=1)
    df = df.fillna(value=10)
    
    return df
