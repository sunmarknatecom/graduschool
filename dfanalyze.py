import os
import pandas as pd
from scipy.stats import shapiro

file_list = [elem for elem in os.listdir() if "result" in elem]
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
__, RT_SKL_PVAL = shapiro(RT_SKL_DF["Max (SUVbw)"])
LT_SKL_MAX_MAX  = LT_SKL_DF["Max (SUVbw)"].max()
LT_SKL_MAX_MEAN = LT_SKL_DF["Max (SUVbw)"].mean()
LT_SKL_MAX_STD  = LT_SKL_DF["Max (SUVbw)"].std()
__, LT_SKL_PVAL = shapiro(LT_SKL_DF["Max (SUVbw)"])
RT_HUM_MAX_MAX  = RT_HUM_DF["Max (SUVbw)"].max()
RT_HUM_MAX_MEAN = RT_HUM_DF["Max (SUVbw)"].mean()
RT_HUM_MAX_STD  = RT_HUM_DF["Max (SUVbw)"].std()
__, RT_HUM_PVAL = shapiro(RT_HUM_DF["Max (SUVbw)"])
LT_HUM_MAX_MAX  = LT_HUM_DF["Max (SUVbw)"].max()
LT_HUM_MAX_MEAN = LT_HUM_DF["Max (SUVbw)"].mean()
LT_HUM_MAX_STD  = LT_HUM_DF["Max (SUVbw)"].std()
__, LT_HUM_PVAL = shapiro(LT_HUM_DF["Max (SUVbw)"])
RT_FEM_MAX_MAX  = RT_FEM_DF["Max (SUVbw)"].max()
RT_FEM_MAX_MEAN = RT_FEM_DF["Max (SUVbw)"].mean()
RT_FEM_MAX_STD  = RT_FEM_DF["Max (SUVbw)"].std()
__, RT_FEM_PVAL = shapiro(RT_FEM_DF["Max (SUVbw)"])
LT_FEM_MAX_MAX  = LT_FEM_DF["Max (SUVbw)"].max()
LT_FEM_MAX_MEAN = LT_FEM_DF["Max (SUVbw)"].mean()
LT_FEM_MAX_STD  = LT_FEM_DF["Max (SUVbw)"].std()
__, LT_FEM_PVAL = shapiro(LT_FEM_DF["Max (SUVbw)"])
METS_MAX_MAX  = METS_DF["Max (SUVbw)"].max()
METS_MAX_MEAN = METS_DF["Max (SUVbw)"].mean()
METS_MAX_STD  = METS_DF["Max (SUVbw)"].std()
__, METS_PVAL = shapiro(METS_DF["Max (SUVbw)"])
DEGS_MAX_MAX  = DEGS_DF["Max (SUVbw)"].max()
DEGS_MAX_MEAN = DEGS_DF["Max (SUVbw)"].mean()
DEGS_MAX_STD  = DEGS_DF["Max (SUVbw)"].std()
__, DEGS_PVAL = shapiro(DEGS_DF["Max (SUVbw)"])

# print("RT_SKULL_MAX", RT_SKL_MAX_MAX, "%.2f"%RT_SKL_MAX_MEAN, "%.2f"%RT_SKL_MAX_STD)
# print("LT_SKULL_MAX", LT_SKL_MAX_MAX, "%.2f"%LT_SKL_MAX_MEAN, "%.2f"%LT_SKL_MAX_STD)
# print("RT_HUMERUS_MAX", RT_HUM_MAX_MAX, "%.2f"%RT_HUM_MAX_MEAN, "%.2f"%RT_HUM_MAX_STD)
# print("LT_HUMERUS_MAX", LT_HUM_MAX_MAX, "%.2f"%LT_HUM_MAX_MEAN, "%.2f"%LT_HUM_MAX_STD)
# print("RT_FEMUR_MAX", RT_FEM_MAX_MAX, "%.2f"%RT_FEM_MAX_MEAN, "%.2f"%RT_FEM_MAX_STD)
# print("LT_FEMUR_MAX", LT_FEM_MAX_MAX, "%.2f"%LT_FEM_MAX_MEAN, "%.2f"%LT_FEM_MAX_STD)
# print("METASTASIS_MAX", METS_MAX_MAX, "%.2f"%METS_MAX_MEAN, "%.2f"%METS_MAX_STD)
# print("DEG_MAX", DEGS_MAX_MAX, "%.2f"%DEGS_MAX_MEAN, "%.2f"%DEGS_MAX_STD)


total_num = len(RT_SKL_DF)+ len(LT_SKL_DF) + len(RT_HUM_DF) + len(LT_HUM_DF) + len(RT_FEM_DF) + len(LT_FEM_DF) + len(METS_DF) + len(DEGS_DF)

# 'Meta'로 시작하는 Contour가 있는 Index_Number 추출

meta_indices = total_df[total_df['Contour'].str.startswith('Meta', na=False)]['Index_Number'].unique()

# Meta가 있는 Index_Number를 포함하는 행만 추출
META_DF = total_df[total_df['Index_Number'].isin(meta_indices)]

# Meta가 없는 Index_Number를 포함하는 행만 추출
NO_META_DF = total_df[~total_df['Index_Number'].isin(meta_indices)]

NO_META_RT_SKL_DF = NO_META_DF[NO_META_DF['Contour'].str.startswith('Rt_s')]
NO_META_LT_SKL_DF = NO_META_DF[NO_META_DF['Contour'].str.startswith('Lt_s')]
NO_META_RT_HUM_DF = NO_META_DF[NO_META_DF['Contour'].str.startswith('Rt._h')]
NO_META_LT_HUM_DF = NO_META_DF[NO_META_DF['Contour'].str.startswith('Lt_h')]
NO_META_RT_FEM_DF = NO_META_DF[NO_META_DF['Contour'].str.startswith('Rt_f')]
NO_META_LT_FEM_DF = NO_META_DF[NO_META_DF['Contour'].str.startswith('Lt_f')]


META_RT_SKL_DF = META_DF[META_DF['Contour'].str.startswith('Rt_s')]
META_LT_SKL_DF = META_DF[META_DF['Contour'].str.startswith('Lt_s')]
META_RT_HUM_DF = META_DF[META_DF['Contour'].str.startswith('Rt._h')]
META_LT_HUM_DF = META_DF[META_DF['Contour'].str.startswith('Lt_h')]
META_RT_FEM_DF = META_DF[META_DF['Contour'].str.startswith('Rt_f')]
META_LT_FEM_DF = META_DF[META_DF['Contour'].str.startswith('Lt_f')]

NO_META_RT_SKL_MAX_MAX  = NO_META_RT_SKL_DF["Max (SUVbw)"].max()
NO_META_RT_SKL_MAX_MEAN = NO_META_RT_SKL_DF["Max (SUVbw)"].mean()
NO_META_RT_SKL_MAX_STD  = NO_META_RT_SKL_DF["Max (SUVbw)"].std()
__, NO_META_RT_SKL_PVAL = shapiro(NO_META_RT_SKL_DF["Max (SUVbw)"])
NO_META_LT_SKL_MAX_MAX  = NO_META_LT_SKL_DF["Max (SUVbw)"].max()
NO_META_LT_SKL_MAX_MEAN = NO_META_LT_SKL_DF["Max (SUVbw)"].mean()
NO_META_LT_SKL_MAX_STD  = NO_META_LT_SKL_DF["Max (SUVbw)"].std()
__, NO_META_LT_SKL_PVAL = shapiro(NO_META_LT_SKL_DF["Max (SUVbw)"])
NO_META_RT_HUM_MAX_MAX  = NO_META_RT_HUM_DF["Max (SUVbw)"].max()
NO_META_RT_HUM_MAX_MEAN = NO_META_RT_HUM_DF["Max (SUVbw)"].mean()
NO_META_RT_HUM_MAX_STD  = NO_META_RT_HUM_DF["Max (SUVbw)"].std()
__, NO_META_RT_HUM_PVAL = shapiro(NO_META_RT_HUM_DF["Max (SUVbw)"])
NO_META_LT_HUM_MAX_MAX  = NO_META_LT_HUM_DF["Max (SUVbw)"].max()
NO_META_LT_HUM_MAX_MEAN = NO_META_LT_HUM_DF["Max (SUVbw)"].mean()
NO_META_LT_HUM_MAX_STD  = NO_META_LT_HUM_DF["Max (SUVbw)"].std()
__, NO_META_LT_HUM_PVAL = shapiro(NO_META_LT_HUM_DF["Max (SUVbw)"])
NO_META_RT_FEM_MAX_MAX  = NO_META_RT_FEM_DF["Max (SUVbw)"].max()
NO_META_RT_FEM_MAX_MEAN = NO_META_RT_FEM_DF["Max (SUVbw)"].mean()
NO_META_RT_FEM_MAX_STD  = NO_META_RT_FEM_DF["Max (SUVbw)"].std()
__, NO_META_RT_FEM_PVAL = shapiro(NO_META_RT_FEM_DF["Max (SUVbw)"])
NO_META_LT_FEM_MAX_MAX  = NO_META_LT_FEM_DF["Max (SUVbw)"].max()
NO_META_LT_FEM_MAX_MEAN = NO_META_LT_FEM_DF["Max (SUVbw)"].mean()
NO_META_LT_FEM_MAX_STD  = NO_META_LT_FEM_DF["Max (SUVbw)"].std()
__, NO_META_LT_FEM_PVAL = shapiro(NO_META_LT_FEM_DF["Max (SUVbw)"])

META_RT_SKL_MAX_MAX  = META_RT_SKL_DF["Max (SUVbw)"].max()
META_RT_SKL_MAX_MEAN = META_RT_SKL_DF["Max (SUVbw)"].mean()
META_RT_SKL_MAX_STD  = META_RT_SKL_DF["Max (SUVbw)"].std()
__, META_RT_SKL_PVAL = shapiro(META_RT_SKL_DF["Max (SUVbw)"])
META_LT_SKL_MAX_MAX  = META_LT_SKL_DF["Max (SUVbw)"].max()
META_LT_SKL_MAX_MEAN = META_LT_SKL_DF["Max (SUVbw)"].mean()
META_LT_SKL_MAX_STD  = META_LT_SKL_DF["Max (SUVbw)"].std()
__, META_LT_SKL_PVAL = shapiro(META_LT_SKL_DF["Max (SUVbw)"])
META_RT_HUM_MAX_MAX  = META_RT_HUM_DF["Max (SUVbw)"].max()
META_RT_HUM_MAX_MEAN = META_RT_HUM_DF["Max (SUVbw)"].mean()
META_RT_HUM_MAX_STD  = META_RT_HUM_DF["Max (SUVbw)"].std()
__, META_RT_HUM_PVAL = shapiro(META_RT_HUM_DF["Max (SUVbw)"])
META_LT_HUM_MAX_MAX  = META_LT_HUM_DF["Max (SUVbw)"].max()
META_LT_HUM_MAX_MEAN = META_LT_HUM_DF["Max (SUVbw)"].mean()
META_LT_HUM_MAX_STD  = META_LT_HUM_DF["Max (SUVbw)"].std()
__, META_LT_HUM_PVAL = shapiro(META_LT_HUM_DF["Max (SUVbw)"])
META_RT_FEM_MAX_MAX  = META_RT_FEM_DF["Max (SUVbw)"].max()
META_RT_FEM_MAX_MEAN = META_RT_FEM_DF["Max (SUVbw)"].mean()
META_RT_FEM_MAX_STD  = META_RT_FEM_DF["Max (SUVbw)"].std()
__, META_RT_FEM_PVAL = shapiro(META_RT_FEM_DF["Max (SUVbw)"])
META_LT_FEM_MAX_MAX  = META_LT_FEM_DF["Max (SUVbw)"].max()
META_LT_FEM_MAX_MEAN = META_LT_FEM_DF["Max (SUVbw)"].mean()
META_LT_FEM_MAX_STD  = META_LT_FEM_DF["Max (SUVbw)"].std()
__, META_LT_FEM_PVAL = shapiro(META_LT_FEM_DF["Max (SUVbw)"])


# 통계값을 딕셔너리로 정리
stats_dict = {
    "Group": [
        "RT_SKL", "LT_SKL", "RT_HUM", "LT_HUM", "RT_FEM", "LT_FEM",
        "METS", "DEGS",
        "NO_META_RT_SKL", "NO_META_LT_SKL", "NO_META_RT_HUM", "NO_META_LT_HUM",
        "NO_META_RT_FEM", "NO_META_LT_FEM",
        "META_RT_SKL", "META_LT_SKL", "META_RT_HUM", "META_LT_HUM",
        "META_RT_FEM", "META_LT_FEM"
    ],
    "Max": [
        RT_SKL_MAX_MAX, LT_SKL_MAX_MAX, RT_HUM_MAX_MAX, LT_HUM_MAX_MAX,
        RT_FEM_MAX_MAX, LT_FEM_MAX_MAX,
        METS_MAX_MAX, DEGS_MAX_MAX,
        NO_META_RT_SKL_MAX_MAX, NO_META_LT_SKL_MAX_MAX, NO_META_RT_HUM_MAX_MAX,
        NO_META_LT_HUM_MAX_MAX, NO_META_RT_FEM_MAX_MAX, NO_META_LT_FEM_MAX_MAX,
        META_RT_SKL_MAX_MAX, META_LT_SKL_MAX_MAX, META_RT_HUM_MAX_MAX,
        META_LT_HUM_MAX_MAX, META_RT_FEM_MAX_MAX, META_LT_FEM_MAX_MAX
    ],
    "Mean": [
        RT_SKL_MAX_MEAN, LT_SKL_MAX_MEAN, RT_HUM_MAX_MEAN, LT_HUM_MAX_MEAN,
        RT_FEM_MAX_MEAN, LT_FEM_MAX_MEAN,
        METS_MAX_MEAN, DEGS_MAX_MEAN,
        NO_META_RT_SKL_MAX_MEAN, NO_META_LT_SKL_MAX_MEAN, NO_META_RT_HUM_MAX_MEAN,
        NO_META_LT_HUM_MAX_MEAN, NO_META_RT_FEM_MAX_MEAN, NO_META_LT_FEM_MAX_MEAN,
        META_RT_SKL_MAX_MEAN, META_LT_SKL_MAX_MEAN, META_RT_HUM_MAX_MEAN,
        META_LT_HUM_MAX_MEAN, META_RT_FEM_MAX_MEAN, META_LT_FEM_MAX_MEAN
    ],
    "STD": [
        RT_SKL_MAX_STD, LT_SKL_MAX_STD, RT_HUM_MAX_STD, LT_HUM_MAX_STD,
        RT_FEM_MAX_STD, LT_FEM_MAX_STD,
        METS_MAX_STD, DEGS_MAX_STD,
        NO_META_RT_SKL_MAX_STD, NO_META_LT_SKL_MAX_STD, NO_META_RT_HUM_MAX_STD,
        NO_META_LT_HUM_MAX_STD, NO_META_RT_FEM_MAX_STD, NO_META_LT_FEM_MAX_STD,
        META_RT_SKL_MAX_STD, META_LT_SKL_MAX_STD, META_RT_HUM_MAX_STD,
        META_LT_HUM_MAX_STD, META_RT_FEM_MAX_STD, META_LT_FEM_MAX_STD
    ],
    "P-Value": [
        RT_SKL_PVAL, LT_SKL_PVAL, RT_HUM_PVAL, LT_HUM_PVAL,
        RT_FEM_PVAL, LT_FEM_PVAL,
        METS_PVAL, DEGS_PVAL,
        NO_META_RT_SKL_PVAL, NO_META_LT_SKL_PVAL, NO_META_RT_HUM_PVAL,
        NO_META_LT_HUM_PVAL, NO_META_RT_FEM_PVAL, NO_META_LT_FEM_PVAL,
        META_RT_SKL_PVAL, META_LT_SKL_PVAL, META_RT_HUM_PVAL,
        META_LT_HUM_PVAL, META_RT_FEM_PVAL, META_LT_FEM_PVAL
    ]
}

# DataFrame으로 변환
stats_df = pd.DataFrame(stats_dict)

stats_df.to_csv("suvbw_statistics_2.csv", index=True)

print("CSV 저장 완료: suvbw_statistics_2.csv")
