import pandas as pd
import numpy as np
import scipy.stats
import altair as alt

# Import raw data
df_healthy = pd.read_csv('https://github.com/BrainPowerChemE/brainpowerdata/blob/main/raw_data/healthy_data.csv?raw=true')
df_PD_MCI_LBD = pd.read_csv('https://github.com/BrainPowerChemE/brainpowerdata/blob/main/raw_data/PD_MCI_LBD_data.csv?raw=true')
df_PD = pd.read_csv('https://github.com/BrainPowerChemE/brainpowerdata/blob/main/raw_data/PD_data.csv?raw=true')
df_AD_MCI = pd.read_csv('https://github.com/BrainPowerChemE/brainpowerdata/blob/main/raw_data/AD_MCI_data.csv?raw=true')