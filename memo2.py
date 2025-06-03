import orbismarci as om
import pandas as pd
import os, copy
import numpy as np

idx_list = os.listdir(".\\data\\")
bone_index = om.get_bone_indices()
organs = om.get_organs()
out_df = []
for idx in [idx_list[197]]:
    print("Starting to process", idx)
    raw_ct_image, raw_lb_image, ct_image, nm_image, suv_nm_image, lb_image = om.get_images(idx)
    print("Loaded images")
    volume = om.get_nm_vol_info(idx)
    temp_dict = {"idx":idx}
    for elem in bone_index:
        temp_lb_image = om.extract_binary_mask_label(lb_image, seg_n = int(elem))
        temp_out_image = om.get_nm_stat_info(suv_nm_image, temp_lb_image)
        print(f"seg: {elem}, volume: {volume*temp_out_image[0]}, min: {temp_out_image[2]}, max: {temp_out_image[1]}, mean: {temp_out_image[3]}, std: {temp_out_image[4]}")
        temp_dict[organs[int(elem)]+"_vol"] = volume * temp_out_image[0]
        temp_dict[organs[int(elem)]+"_min"] = temp_out_image[2]
        temp_dict[organs[int(elem)]+"_max"] = temp_out_image[1]
        temp_dict[organs[int(elem)]+"_mean"] = temp_out_image[3]
        temp_dict[organs[int(elem)]+"_std"] = temp_out_image[4]
    out_df.append(temp_dict)
    print("Finished processing", idx)

result_df = pd.DataFrame(out_df)


ct_path, nm_path, lb_path = om.get_file_paths("210")
ct_objs = om.open_CT_obj(ct_path)
nm_obj = om.open_NM_obj(nm_path)
ct_image = om.load_CT_image(ct_objs)
nm_image = om.load_NM_image(nm_obj)
lb_image = om.load_LB_image(lb_path)
raw_ct_image, raw_lb_image, ct_image, nm_image, suv_nm_image, lb_image = om.get_images("210")

# range manupulation

axial_range = om.find_sig_lb_index_range(lb_image)
sagital_range = om.find_axial_sig_lb_index_range(lb_image, axial_index=1)
coronal_range = om.find_axial_sig_lb_index_range(lb_image, axial_index=2)



import os

folder_list = sorted(os.listdir())

def path_cleaning():
    grouped_index_dict = {}
    for folder_name in folder_list:
        file_names = os.listdir(folder_name)  # 해당 폴더 내 파일 리스트
        file_paths = [os.path.join(folder_name, file_name) for file_name in file_names]
        grouped_index_dict[folder_name] = file_paths
    return grouped_index_dict
    # grouped_index_dict  {"001": ["001_ref.txt", "001_deg.txt", "001_meta.txt"], "002": ["002_ref.txt", "002_deg.txt", "002_meta.txt"]}

def reference_result_cleaning(grouped_index_dict_list):
    """
    input : grouped_index_dict_list ['001\\001_ref.txt', '001\\001_deg.txt', '001\\001_meta.txt']
    output : dict {"Rt_skull_max": value, "Rt_skull_min": value, ..., "Deg_Rt_...": value, ... , "Meta_organ": value,...}
    """
    grouped_result_dict = {}
    for idx, paths in grouped_index_dict_list.items():
        temp_out_dict = {}
        for path_ in paths:
            try:
                if "ref" in os.path.basename(path_):
                    with open(path_, "r") as file:
                        lines = [line.strip() for line in file.readlines() if line.strip()]  # Remove empty lines
                    for i in range(len(lines)):
                        if i%6 == 0:
                            name = "Ref_" + lines[i]
                        elif i%6 == 2:
                            temp_out_dict[name+"_"+lines[i].split(" ")[0]] = float(lines[i].split(" ")[1])
                        elif i%6 == 3:
                            temp_out_dict[name+"_"+lines[i].split(" ")[0]] = float(lines[i].split(" ")[1])
                        elif i%6 == 5:
                            temp_out_dict[name+"_"+lines[i].split(" ")[0]] = float(lines[i].split(" ")[1])
                        else:
                            continue
                else:
                    continue
            except:
                print("Error in ", idx, path_)
        grouped_result_dict[idx] = temp_out_dict
    return grouped_result_dict

def deg_result_cleaning(grouped_index_dict_list):
    """
    input : grouped_index_dict_list ['001\\001_ref.txt', '001\\001_deg.txt', '001\\001_meta.txt']
    output : dict {"Rt_skull_max": value, "Rt_skull_min": value, ..., "Deg_Rt_...": value, ... , "Meta_organ": value,...}
    """
    grouped_result_dict = {}
    for idx, paths in grouped_index_dict_list.items():
        temp_out_dict = {}
        for path_ in paths:
            try:
                if "deg" in os.path.basename(path_):
                    # print(path_)
                    with open(path_, "r") as file:
                        lines = [line.strip() for line in file.readlines() if line.strip()]  # Remove empty lines
                    for i in range(len(lines)):
                        if i%6 == 0:
                            name = "Deg_"+str(int(i//6))+"_"+str(int(len(lines)//6)-1)
                            temp_out_dict[name+"loc"] = lines[i]
                        elif i%6 == 2:
                            temp_out_dict[name+"_"+lines[i].split(" ")[0]] = float(lines[i].split(" ")[1])
                        elif i%6 == 3:
                            temp_out_dict[name+"_"+lines[i].split(" ")[0]] = float(lines[i].split(" ")[1])
                        elif i%6 == 5:
                            temp_out_dict[name+"_"+lines[i].split(" ")[0]] = float(lines[i].split(" ")[1])
                        else:
                            continue
                else:
                    print("Not a def file: ", path_)
                    continue
            except:
                print("Error in ", idx, path_)
        grouped_result_dict[idx] = temp_out_dict
    return grouped_result_dict

def meta_result_cleaning(grouped_index_dict_list):
    """
    input : grouped_index_dict_list ['001\\001_ref.txt', '001\\001_deg.txt', '001\\001_meta.txt']
    output : dict {"Rt_skull_max": value, "Rt_skull_min": value, ..., "Deg_Rt_...": value, ... , "Meta_organ": value,...}
    """
    grouped_result_dict = {}
    for idx, paths in grouped_index_dict_list.items():
        temp_out_dict = {}
        for path_ in paths:
            try:
                if "meta" in os.path.basename(path_):
                    # print(path_)
                    with open(path_, "r") as file:
                        lines = [line.strip() for line in file.readlines() if line.strip()]  # Remove empty lines
                    for i in range(len(lines)):
                        if i%6 == 0:
                            name = "Meta_"+str(int(i//6))+"_"+str(int(len(lines)//6)-1)
                            temp_out_dict[name+"loc"] = lines[i]
                        elif i%6 == 2:
                            temp_out_dict[name+"_"+lines[i].split(" ")[0]] = float(lines[i].split(" ")[1])
                        elif i%6 == 3:
                            temp_out_dict[name+"_"+lines[i].split(" ")[0]] = float(lines[i].split(" ")[1])
                        elif i%6 == 5:
                            temp_out_dict[name+"_"+lines[i].split(" ")[0]] = float(lines[i].split(" ")[1])
                        else:
                            continue
                else:
                    print("Not a meta file: ", path_)
                    continue
            except:
                print("Error in ", idx, path_)
        grouped_result_dict[idx] = temp_out_dict
    return grouped_result_dict

paths = path_cleaning()
out_dict = reference_result_cleaning(paths)
deg_out_dict = deg_result_cleaning(paths)
meta_out_dict = meta_result_cleaning(paths)

for k, v in out_dict.items():
    print(k, len(v.keys()))

for k, v in deg_out_dict.items():
    print(k, len(v.keys()))

for k, v in meta_out_dict.items():
    print(k, len(v.keys()))

import sys

sys.path.append("D:\\99Gradu\\")

import orbismarci as om
import pandas as pd

paths = om.path_cleaning()
ref_out_dict = om.reference_result_cleaning(paths)
deg_out_dict = om.deg_result_cleaning(paths)
meta_out_dict = om.meta_result_cleaning(paths)

ref_df = pd.DataFrame.from_dict(ref_out_dict, orient='index')
deg_df = pd.DataFrame.from_dict(deg_out_dict, orient='index')
meta_df = pd.DataFrame.from_dict(meta_out_dict, orient='index')

ref_df.to_csv("ref_results.csv", index=True)
deg_df.to_csv("deg_results.csv", index=True)
meta_df.to_csv("meta_results.csv", index=True)
