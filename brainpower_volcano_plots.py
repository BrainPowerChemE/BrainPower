import numpy as np
import pandas as pd
import scipy.stats
import altair as alt




def clean_raw_bp_data(frames, names, retglobals=False):
    # Returns clean_frame1, clean_frame2, clean_frame3, clean_frame4
    # retglobals (default=true) also returns list_protein_long, list_protein_short, dict_proteins for global use without calling bp_altair_util.[global variable]

    # Set index of frames to assay id
    for frame in frames:
        frame.set_index('group', inplace=True)

    # Confirm headers (long form protein name with peptide id) of all the dataframes are the same
    for frame in frames:
        assert all(frame.columns == frames[0].columns), 'column headers are not equal' 

    # List column headers AKA protein name long forms. Frame[0] chosen arbitrarily. the assertion on line 17 should ensure they are redundant. Set global variable.
    global list_protein_long
    list_protein_long = list(frames[0].columns)

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
    newframes = []
    for i in range(0,len(frames)):
        newframes.append(frames[i].rename(columns=dict_proteins))

    for i in range(0,len(newframes)):
        newframes[i].name = names[i]

        

    # Reverse dictionary for future use. Keys are shorthand protein id, which now matches column headers
    dict_proteins = dict([t[::-1] for t in list(dict_proteins.items())])    
    
    # Return clean frames
    if retglobals == False:
        return newframes
    if retglobals == True:
        return newframe1, newframe2, newframe3, newframe4, list_protein_long, list_protein_short, dict_proteins


def gen_volcano_frames(frames_tuple, names, equalvar=False, nanpolicy='omit',neglog10p_sig_cutoff=1.3):
    def gen_volcano_tuple(frames_tuple, equalvar=False, nanpolicy='omit'):
        # Returns ['protein','avg_expr_cond','avg_expr_healthy','log2_FC','t_value','log10_p_value']
        datalists = []
        for i in range(0,len(frames_tuple)):
            datalist = []
            for protein in list_protein_short:
                t_stat, p_stat = scipy.stats.ttest_ind(
                    frames_tuple[i][0][protein],
                    frames_tuple[i][1][protein],
                    equal_var=equalvar, 
                    nan_policy=nanpolicy
                    )
            
                datalist.append(
                    [protein,frames_tuple[i][0][protein].mean(), frames_tuple[i][1][protein].mean(),
                            np.log2(frames_tuple[i][0][protein].mean()/frames_tuple[i][1][protein].mean()),
                            float(t_stat), float(np.log10(p_stat))*-1]
                    )
            datalists.append(datalist)
        return datalists

    def gen_volcano_frame(datalists, names):
        volcano_frames = []
        for i in range(0,len(datalists)):
            volcano_frame = pd.DataFrame(data=datalists[i], columns=['protein','avg_expr_cond','avg_expr_healthy','log2_FC','t_value','-log10_p_value'])
            volcano_frame.name = names[i]
            volcano_frames.append(volcano_frame)
        return volcano_frames

    volcano_frames = gen_volcano_frame(gen_volcano_tuple(frames_tuple, equalvar, nanpolicy), names)

    def gen_pandas_volcano_significance_column(frame,neglog10p_sig_cutoff=1.3):
        significance_categories = ['nosig', 'significant_downreg', 'significant_upreg']
        for i in range(0,len(volcano_frames)):
            significance_conditions = [
            (volcano_frames[i]['-log10_p_value'] < neglog10p_sig_cutoff),
            (volcano_frames[i]['-log10_p_value'] > neglog10p_sig_cutoff) & (volcano_frames[i]['log2_FC'] < 0),
            (volcano_frames[i]['-log10_p_value'] > neglog10p_sig_cutoff) & (volcano_frames[i]['log2_FC'] > 0)
            ]
            volcano_frames[i]['significance'] = np.select(significance_conditions,significance_categories)
        return volcano_frames

    volcano_frames = gen_pandas_volcano_significance_column(volcano_frames)
    return volcano_frames

def gen_altair_volcano_plots(volcano_frames):
    significance_categories = ['nosig', 'significant_downreg', 'significant_upreg']
    color_range = ['grey','blue','red']
    charts = []
    for i in range(0,len(volcano_frames)):
        chart = alt.Chart(volcano_frames[i], title=volcano_frames[i].name).mark_point().encode(
        x='log2_FC',
        y='-log10_p_value',
        tooltip='protein',
        color=alt.Color('significance', scale=alt.Scale(domain=significance_categories, range=color_range))).interactive()
        charts.append(chart)
    return charts


def generate_volcano_ready_dataframes(ret_whole_data_frames=False,ret_whole_volcano_frames=False):
    """if ret_whole_data_frames == True and ret_whole_volcano_frames == True:
        return df_AD_MCI, df_healthy, df_PD, df_PD_MCI_LBD, df_AD_MCI_volcano_ready, df_PD_volcano_ready, df_PD_MCI_volcano_ready, df_PD_MCI_vs_PD_volcano_ready

    elif ret_whole_data_frames == True and ret_whole_volcano_frames == False:
        return df_AD_MCI, df_healthy, df_PD, df_PD_MCI_LBD, volcano_frames
        
    elif ret_whole_data_frames == False and ret_whole_volcano_frames== False:
        return frames, volcano_frames
    """
    df_AD_MCI = pd.read_csv('https://github.com/BrainPowerChemE/brainpowerdata/blob/main/raw_data/AD_MCI_data.csv?raw=true')
    df_healthy = pd.read_csv('https://github.com/BrainPowerChemE/brainpowerdata/blob/main/raw_data/healthy_data.csv?raw=true')
    df_PD = pd.read_csv('https://github.com/BrainPowerChemE/brainpowerdata/blob/main/raw_data/PD_data.csv?raw=true')
    df_PD_MCI_LBD = pd.read_csv('https://github.com/BrainPowerChemE/brainpowerdata/blob/main/raw_data/PD_MCI_LBD_data.csv?raw=true')
    
    df_AD_MCI, df_healthy, df_PD, df_PD_MCI_LBD = clean_raw_bp_data(
        frames=[df_AD_MCI, df_healthy, df_PD, df_PD_MCI_LBD],
        names = [
            'Alzheimers',
            'Healthy',
            'Parkinsons',
            'Parkinsons_MCI'],
       retglobals=False)

    volcano_frames = gen_volcano_frames(
    frames_tuple=[[df_AD_MCI,df_healthy],[df_PD,df_healthy],[df_PD_MCI_LBD,df_healthy],[df_PD_MCI_LBD,df_PD]],
    names = [
        'Proteins in CSF of Alzheimers patients compared to healthy patients',
        'Proteins in CSF of Parkinsons patients compared to healthy patients',
        'Proteins in CSF of Parkinsons patients with mild cognitive impairment compared to healthy patients',
        'Proteins in CSF of Parkinsons patients with mild cognitive impairment compared to PD patients without MCI'],
    equalvar=False,
    nanpolicy='omit'
    )

    df_AD_MCI_volcano_ready = volcano_frames[0]
    df_PD_volcano_ready = volcano_frames[1]
    df_PD_MCI_volcano_ready = volcano_frames[2]
    df_PD_MCI_vs_PD_volcano_ready = volcano_frames[3]

    frames = [df_AD_MCI, df_healthy, df_PD, df_PD_MCI_LBD]

    if ret_whole_data_frames == True and ret_whole_volcano_frames == True:
        print('Returned df_AD_MCI, df_healthy, df_PD, df_PD_MCI_LBD, df_AD_MCI_volcano_ready, df_PD_volcano_ready, df_PD_MCI_volcano_ready, df_PD_MCI_vs_PD_volcano_ready')
        return df_AD_MCI, df_healthy, df_PD, df_PD_MCI_LBD, df_AD_MCI_volcano_ready, df_PD_volcano_ready, df_PD_MCI_volcano_ready, df_PD_MCI_vs_PD_volcano_ready
    elif ret_whole_data_frames == True and ret_whole_volcano_frames == False:
        print('Returned df_AD_MCI, df_healthy, df_PD, df_PD_MCI_LBD, volcano_frames')
        return df_AD_MCI, df_healthy, df_PD, df_PD_MCI_LBD, volcano_frames
    elif ret_whole_data_frames == False and ret_whole_volcano_frames== False:
        print('Returned frames, volcano_frames')
        return frames, volcano_frames