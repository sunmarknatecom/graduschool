import os
import pandas as pd
from scipy.stats import shapiro

file_list = os.listdir()
df_list = []
for file_path in file_list:
    temp_index = file_path[:3]
    df = pd.read_csv(file_path)
    df['Index_Number'] = temp_index
    df_list.append(df)

total_df = pd.concat(df_list, ignore_index = True)

RT_SKL_DF = total_df[total_df['Contour'].str.startswith('Rt_s')]
LT_SKL_DF = total_df[total_df['Contour'].str.startswith('Lt_s')]
RT_HUM_DF = total_df[total_df['Contour'].str.startswith('Rt._h')]
LT_HUM_DF = total_df[total_df['Contour'].str.startswith('Lt_h')]
RT_FEM_DF = total_df[total_df['Contour'].str.startswith('Rt_f')]
LT_FEM_DF = total_df[total_df['Contour'].str.startswith('Lt_f')]

METS_DF = total_df[total_df['Contour'].str.startswith('Met')]
DEGS_DF= total_df[total_df['Contour'].str.startswith('Deg')]

COL_INDEX = total_df.columns

# COL_INDEX=
"""
00 Contour
01 Finding
02 Series Date
03 Integral Total (SUVbw*ml)
04 Kurtosis (-)
05 Max (SUVbw)
06 Mean (SUVbw)
07 Median (SUVbw)
08 Min (SUVbw)
09 Skewness (-)
10 Slice with Maximum Value (#)
11 Sphere Diameter (cm)
12 Standard Deviation (SUVbw)
13 Total (SUVbw)
14 Volume (ml)
15 Index_Number
"""

RT_SKL_MAX_MAX  = RT_SKL_DF["Max (SUVbw)"].max()
RT_SKL_MAX_MEAN = RT_SKL_DF["Max (SUVbw)"].mean()
RT_SKL_MAX_STD  = RT_SKL_DF["Max (SUVbw)"].std()

print(RT_SKL_MAX_MAX, RT_SKL_MAX_MEAN, RT_SKL_MAX_STD)

