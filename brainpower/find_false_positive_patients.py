import pandas as pd
import numpy as np

def find_false_positive_patients(metadata, ml_results): 
    """
    Function: identifies the "healthy" patients that were identified as diseased by the ML model
    
    Input: dataframe (ml_results.df) with predicted condition vs. actual condition and Metadata.csv contains patient information
    
    Output: DataFrame (false_positives_info.df) of the "healthy" patients that were identified as diseased by the ML model
    """
    
    false_positives=ml_results.iloc[np.where((ml_results['Actual'] == 'Healthy') & ~(ml_results['Predicted'] == 'Healthy'))]

    false_positives_info=[]
    false_positives_IDs=list(false_positives['patient_ID'])
    for ID in false_positives_IDs: 
        false_positives_info.append(metadata.iloc[np.where(metadata['Public Sample ID']==ID)])
    false_positives_info = pd.concat(false_positives_info)
    
    return false_positives_info
