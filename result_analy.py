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
