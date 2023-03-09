import pandas as pd
import numpy as np
import sklearn.model_selection


def split_cats_by_tolerance(frame,tolerance,silent=False,randomstate=None,split=0.15,step=1,categories=['Healthy','AD_MCI','PD','PD_MCI_LBD']):
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
            
    return df_dev, df_test, randomstate-1
