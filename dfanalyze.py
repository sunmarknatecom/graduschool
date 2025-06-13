import os
import pandas as pd

file_list = os.listdir()
df_list = []
for file_path in file_list:
    temp_index = file_path[:3]
    df = pd.read_csv(file_path)
    df['Index_Number'] = temp_index
    df_list.append(df)

total_df = pd.concat(df_list, ignore_index = True)

RT_SKULL_DF = total_df[total_df['Contour'].str.startswith('Rt_s')]
LT_SKULL_DF = total_df[total_df['Contour'].str.startswith('Lt_s')]
RT_HUMERUS_DF = total_df[total_df['Contour'].str.startswith('Rt._h')]
LT_HUMERUS_DF = total_df[total_df['Contour'].str.startswith('Lt_h')]
RT_FEMUR_DF = total_df[total_df['Contour'].str.startswith('Rt_f')]
LT_FEMUR_DF = total_df[total_df['Contour'].str.startswith('Lt_f')]

METS = total_df[total_df['Contour'].str.startswith('Met')]
DEGS = total_df[total_df['Contour'].str.startswith('Deg')]
