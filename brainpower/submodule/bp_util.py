import numpy as np
import pandas as pd
import scipy.stats
import altair as alt
import sklearn.preprocessing
import sklearn.model_selection


def reversed_delimited_tuple(string,delimiter='|'):
    delimited_tuple = string.split(delimiter)
    reversed_tuple = delimited_tuple[::-1]
    return reversed_tuple

def clean_raw_bp_data(frame,retdict=True):
    # Returns clean_frame1, clean_frame2, clean_frame3, clean_frame4
    # retglobals (default=true) also returns list_protein_long, list_protein_short, dict_proteins for global use without calling bp_altair_util.[global variable]

    # Set index of frames to assay id
    

    # List column headers AKA protein name long forms. Frame[0] chosen arbitrarily. the assertion on line 17 should ensure they are redundant. Set global variable.
    global list_protein_long
    list_protein_long = list(frame.columns)

    # function only used in this instance
    def reversed_delimited_tuple(string,delimiter='|'):
        delimited_tuple = string.split(delimiter)
        reversed_tuple = delimited_tuple[::-1]
        return reversed_tuple

    # List proteins by the final short-name identifier used in the column header AKA protein name long form. Set global variable
    global list_protein_short
    list_protein_short = []
    for protein_long in list_protein_long:
        list_protein_short.append(reversed_delimited_tuple(protein_long)[0]) 

    # Confirm this list is unique
    for elem in list_protein_short:
        assert list_protein_short.count(elem) == 1, print(elem,'is not unique')

    # Create dictionary of extended protein info : short identifier. Set global variable
    global dict_proteins
    dict_proteins = {}
    for i in range(0,len(list_protein_short)):
        dict_proteins[list_protein_long[i]] = list_protein_short[i]

    # Write clean data frames
    newframe = frame.rename(columns=dict_proteins)
    

    # Reverse dictionary for future use. Keys are shorthand protein id, which now matches column headers
    dict_proteins = dict([t[::-1] for t in list(dict_proteins.items())])    
    
    # Return clean frames
    if retdict == False:
        return newframe
    if retdict == True:
        return newframe, dict_proteins

def handle_scale_and_nan(frame,nandecision='drop',scale='MinMax'):
    features = list(frame.select_dtypes(include='float64'))
    cat = list(frame.select_dtypes(include='object'))

    if scale == 'MinMax':
        scaler = sklearn.preprocessing.MinMaxScaler().fit(frame[features])
    elif scale == 'Standard':
        scaler = sklearn.preprocessing.StandardScaler().fit(frame[features])
        
    df_cont = pd.DataFrame(data=scaler.transform(frame[features]), columns=features)
    df_cat = pd.DataFrame(data=frame[cat], columns=cat)
    
    frame = pd.concat([df_cat,df_cont],axis=1)
    
    if nandecision == 'mean':
            for feature in features:
                frame[feature].fillna((frame[feature].mean()), inplace=True)
    elif nandecision == 'drop':
            frame = frame.dropna(axis=1)
        
    return frame


def split_cats_by_tolerance(frame,tolerance,
                       dev_test_split=0.15,train_val_split=0.15,
                       randstate=1,step=1,
                       categories=['Healthy','AD_MCI','PD','PD_MCI_LBD'], retdev=False):
    """
    Split a chosen pandas dataframe into train, val, and test.
    df_full: the full dataframe
    tolerance: target standard deviation between category abundances in train, val, test frames.
                EG if category "a" has abundances of [0.554, 0.555, 0.571] in the train, val, and test frames, 
                    the std for "a" is 0.00779. 
                    If the stated tolerance is greater than 0.00779, category a is satisfied. 
                    All categories must be satisfied simultaneously by a single random state to break the loop.
    dev_test_split: first split between dev and test frames
    train_val_split: second split between train and val frames
    initial_random_state: start random state search here
    step: step of random state each loop
    categories: categories present in data
    retdev: if False (default), return = df_train, df_val, df_test
        if True, return = df_train, df_val, df_test, df_dev
    """
    
    import sklearn.model_selection
    import pandas as pd
    tolerable_list = []
    while sum(tolerable_list) != 4:
        standdevs = []
        df_dev, df_test = sklearn.model_selection.train_test_split(frame,test_size=dev_test_split,random_state=randstate)
        df_train, df_val = sklearn.model_selection.train_test_split(df_dev,test_size=train_val_split,random_state=randstate)
        
        train_dict = dict(df_train['group'].value_counts())
        val_dict = dict(df_val['group'].value_counts())
        test_dict = dict(df_test['group'].value_counts())

        tolerable_list = []
        stats_dict = {}
        for i in range(0,len(categories)):
            try:
                train_dict[categories[i]]
                val_dict[categories[i]]
                test_dict[categories[i]]
            except KeyError:
                break
            percents = [
                (train_dict[categories[i]]/len(df_train)),
                (val_dict[categories[i]]/len(df_val)),
                (test_dict[categories[i]]/len(df_test)),
                    ]
            standdev = np.std(percents)
            standdevs.append(standdev)
            if standdev <= tolerance:
                tolerable_list.append(1)
                stats_dict[str(categories[i])] = [[*percents],standdev]
            else:
                tolerable_list.append(0) 
        randstate += step
    
    if sum(tolerable_list) == 4:
#        print(standdevs)
        print('Random state meeting tolerance threshold:',randstate-1)
#        print('Value counts in this state')
#        print(train_dict)
#        print(val_dict)
#        print(test_dict)
#        print()
#        for i in range(0,len(categories)):
#            print('\nPercent',categories[i],'in train, val, test:',stats_dict[categories[i]][0],
#                  '\nStandard deviation of these values:',stats_dict[categories[i]][1],'\n')
    
    if retdev == False:
        return df_train, df_val, df_test, randstate-1, standdevs
    if retdev == True:
        return df_train, df_val, df_test, df_dev, randstate-1, standdevs


def train_cat_equalizer(df_train,randomstate=1,categories=['Healthy','AD_MCI','PD','PD_MCI_LBD']):
    """
    Randomly selects rows among each category such that each category has an equal number of rows.
    Ensures the truncated training frame has the same columns with NaN so if NaN columns are dropped, the truncated training frame has the same features as the val and test frames
    """
      
    # Determine the value counts of each category in in the training datal
    d = dict(df_train['group'].value_counts())
    min_cat = min(d, key=d.get)
    min_count = min(d.values())
    #print('Minimum category,',min_cat,', has',min_count,'rows... returning training data with',min_count,'rows for each category')
    
    # pd.sample will randomly sample rows. All chosen rows must have 
    
    dataframes = []
    for i in range(0,len(categories)):
        dataframes.append(df_train[df_train['group'] == categories[i]].sample(n=min_count,random_state=randomstate))
    trunc_df = pd.concat(dataframes)
    trunc_df = pd.DataFrame(trunc_df)

    return trunc_df

def colorizer(group):
    if group == 'Healthy':
        return 'green'
    if group == 'AD_MCI':
        return 'blue'        
    if group == 'PD':
        return 'red'        
    if group == 'PD_MCI_LBD':
        return 'orange'