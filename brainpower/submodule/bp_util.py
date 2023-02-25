import numpy as np
import pandas as pd
import scipy.stats
import altair as alt




def clean_raw_bp_data(frame,retdict=True):
    # Returns clean_frame1, clean_frame2, clean_frame3, clean_frame4
    # retglobals (default=true) also returns list_protein_long, list_protein_short, dict_proteins for global use without calling bp_altair_util.[global variable]

    # Set index of frames to assay id
    frame.set_index('group', inplace=True)

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