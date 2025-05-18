import os, copy, cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap
from omfileutil import get_file_paths, open_CT_obj, open_NM_obj, load_CT_image, load_LB_image, load_NM_image, load_suv_nm_image
from arrayprocess import get_align_info, realign_ct_image, realign_nm_image, realign_lb_image, transform_ct_image, transform_single_label, transform_label
from imageprocess import cvt_mono_image, cvt_color_image, cvt_mono_image, cvt_color_image
from labelprocess import label_info, full_label_info, extract_raw_mask_label, extract_binary_mask_label, find_min_max_index, blend_images, find_sig_index, find_sig_frame, merge_lb_image
from display import single_channel_view, multi_channel_view, merged_view
from omstat import get_nm_vol_info, get_nm_stat_info

# global variables

# file utils

# array proceesing

# image processing

# label processing

# visualization

def get_images(idx):
    '''
    7 objects
    ---------------------------------------------------------------------------
    NAME                  SIZE       Length
    ---------------------------------------------------------------------------
    1. raw_ct_image(np)   512 X 512  small
    2. raw_lb_image(np)   512 X 512  small (=same raw_ct_image)
    3. tr_ct_image(np)    256 X 256  small (=same raw_ct_image)
    4. nm_image(np)       256 X 256  long
    5. suv_nm_image(np)   256 X 256  long
    6. rn_tr_lb_image(np) 256 X 256  long  (reshape from nm_image)
    ---------------------------------------------------------------------------
    Combination
    1 and 2 (future)
    5 and 6 (present)
    '''
    bones_index = get_bone_indices()
    temp_ct_path, temp_nm_path, temp_lb_path = get_file_paths(idx)
    temp_ct_objs = open_CT_obj(temp_ct_path)
    temp_nm_obj = open_NM_obj(temp_nm_path)
    temp_nm_image = load_NM_image(temp_nm_obj)
    temp_suv_nm_image = load_suv_nm_image(temp_nm_obj)
    temp_lb_image = load_LB_image(temp_lb_path)
    raw_temp_ct_image, tr_temp_ct_image = transform_ct_image(temp_ct_objs, temp_nm_obj)
    # to raw data
    realign_vars = get_align_info(temp_ct_objs, temp_nm_obj)
    nm_start_index = realign_vars["Start ID of NM"]
    lb_start_index = realign_vars["Start ID of CT"]
    nm_end_index = realign_vars["End ID of NM"]
    temp_skip_list = realign_vars["nm_indices_to_exclude"]
    tr_temp_lb_image = merge_lb_image(temp_ct_objs, temp_nm_obj, temp_lb_image, bones_index)
    rn_tr_lb_image = realign_lb_image(temp_nm_image,tr_temp_lb_image, nm_start_index, nm_end_index , temp_skip_list)
    return raw_temp_ct_image, temp_lb_image, tr_temp_ct_image, temp_nm_image, temp_suv_nm_image, rn_tr_lb_image



def check_index(limit_num=10):
    """
    Check the index of the images in the data directory.
    Args:
        limit_num (int): The number of images to check. Default is 10.
    Returns:
        int: 0 if successful, 1 if failed.
    """
    idx_list = os.listdir(".\\data\\")
    for i, elem in enumerate(idx_list[:limit_num]):
        if i == 0:
            print("IDX  CT,             LB,             TR_CT,          NM,             SUV_NM,         RN_TR_LB,      ELEM CHECK")
            raw_ct_image, raw_lb_image, tr_ct_image, raw_nm_image, suv_nm_image, rn_tr_lb_image = get_images(elem)
            basic_unique = np.unique(rn_tr_lb_image)
            check_result = np.all(basic_unique == np.unique(rn_tr_lb_image))
            print(elem, np.shape(raw_ct_image), np.shape(raw_lb_image), np.shape(tr_ct_image), np.shape(raw_nm_image), np.shape(suv_nm_image), np.shape(rn_tr_lb_image), check_result)
        else:
            raw_ct_image, raw_lb_image, tr_ct_image, raw_nm_image, suv_nm_image, rn_tr_lb_image = get_images(elem)
            try:
                check_result = np.all(basic_unique == np.unique(rn_tr_lb_image))
            except:
                check_result = False
            print(elem, np.shape(raw_ct_image), np.shape(raw_lb_image), np.shape(tr_ct_image), np.shape(raw_nm_image), np.shape(suv_nm_image), np.shape(rn_tr_lb_image), check_result)
    return 0



def get_bone_indices():
    """
    Get the bone indices from the organs dictionary.
    Returns:
        list: A list of bone indices.
    """
    return [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117]



def get_organs():
    """
    Get the organs dictionary.
    Returns:
        dict: A dictionary of organs with their corresponding indices.
    """
    return {1 : "spleen", 2 : "kidney_right", 3 : "kidney_left", 4 : "gallbladder", 5 : "liver", 6 : "stomach", 7 : "pancreas", 8 : "adrenal_gland_right", 9 : "adrenal_gland_left", 10 : "lung_upper_lobe_left", 11 : "lung_lower_lobe_left", 12 : "lung_upper_lobe_right", 13 : "lung_middle_lobe_right", 14 : "lung_lower_lobe_right", 15 : "esophagus", 16 : "trachea", 17 : "thyroid_gland", 18 : "small_bowel", 19 : "duodenum", 20 : "colon", 21 : "urinary_bladder", 22 : "prostate", 23 : "kidney_cyst_left", 24 : "kidney_cyst_right", 25 : "sacrum", 26 : "vertebrae_S1", 27 : "vertebrae_L5", 28 : "vertebrae_L4", 29 : "vertebrae_L3", 30 : "vertebrae_L2", 31 : "vertebrae_L1", 32 : "vertebrae_T12", 33 : "vertebrae_T11", 34 : "vertebrae_T10", 35 : "vertebrae_T9", 36 : "vertebrae_T8", 37 : "vertebrae_T7", 38 : "vertebrae_T6", 39 : "vertebrae_T5", 40 : "vertebrae_T4", 41 : "vertebrae_T3", 42 : "vertebrae_T2", 43 : "vertebrae_T1", 44 : "vertebrae_C7", 45 : "vertebrae_C6", 46 : "vertebrae_C5", 47 : "vertebrae_C4", 48 : "vertebrae_C3", 49 : "vertebrae_C2", 50 : "vertebrae_C1", 51 : "heart", 52 : "aorta", 53 : "pulmonary_vein", 54 : "brachiocephalic_trunk", 55 : "subclavian_artery_right", 56 : "subclavian_artery_left", 57 : "common_carotid_artery_right", 58 : "common_carotid_artery_left", 59 : "brachiocephalic_vein_left", 60 : "brachiocephalic_vein_right", 61 : "atrial_appendage_left", 62 : "superior_vena_cava", 63 : "inferior_vena_cava", 64 : "portal_vein_and_splenic_vein", 65 : "iliac_artery_left", 66 : "iliac_artery_right", 67 : "iliac_vena_left", 68 : "iliac_vena_right", 69 : "humerus_left", 70 : "humerus_right", 71 : "scapula_left", 72 : "scapula_right", 73 : "clavicula_left", 74 : "clavicula_right", 75 : "femur_left", 76 : "femur_right", 77 : "hip_left", 78 : "hip_right", 79 : "spinal_cord", 80 : "gluteus_maximus_left", 81 : "gluteus_maximus_right", 82 : "gluteus_medius_left", 83 : "gluteus_medius_right", 84 : "gluteus_minimus_left", 85 : "gluteus_minimus_right", 86 : "autochthon_left", 87 : "autochthon_right", 88 : "iliopsoas_left", 89 : "iliopsoas_right", 90 : "brain", 91 : "skull", 92 : "rib_left_1", 93 : "rib_left_2", 94 : "rib_left_3", 95 : "rib_left_4", 96 : "rib_left_5", 97 : "rib_left_6", 98 : "rib_left_7", 99 : "rib_left_8", 100 : "rib_left_9", 101 : "rib_left_10", 102 : "rib_left_11", 103 : "rib_left_12", 104 : "rib_right_1", 105 : "rib_right_2", 106 : "rib_right_3", 107 : "rib_right_4", 108 : "rib_right_5", 109 : "rib_right_6", 110 : "rib_right_7", 111 : "rib_right_8", 112 : "rib_right_9", 113 : "rib_right_10", 114 : "rib_right_11", 115 : "rib_right_12", 116 : "sternum", 117 : "costal_cartilages"}



if __name__ == "__main__":
    idx = "025"
    bone_index = get_bone_indices()
    organs = get_organs()
    ct_path, nm_path, lb_path = get_file_paths(idx)
    print(ct_path, nm_path, lb_path)
    ct_objs = open_CT_obj(ct_path)
    nm_obj = open_NM_obj(nm_path)
    nm_image = load_NM_image(nm_obj)
    raw_ct_image, raw_lb_image, ct_image, nm_image, suv_nm_image, lb_image = get_images(idx)
    print("Loaded images")
    volume = get_nm_vol_info(idx)
    temp_dict = {"idx":idx}
    for elem in bone_index:
        temp_lb_image = extract_binary_mask_label(lb_image, seg_n = elem)
        temp_out_image = get_nm_stat_info(suv_nm_image, temp_lb_image)
        print(f"elem: {elem}, volu: {volume*temp_out_image[0]}, min: {temp_out_image[2]}, max: {temp_out_image[1]}, mean: {temp_out_image[3]}, std: {temp_out_image[4]}")
        temp_dict[organs[int(elem)]+"_vol"] = volume * temp_out_image[0]
        temp_dict[organs[int(elem)]+"_min"] = temp_out_image[2]
        temp_dict[organs[int(elem)]+"_max"] = temp_out_image[1]
        temp_dict[organs[int(elem)]+"_mean"] = temp_out_image[3]
        temp_dict[organs[int(elem)]+"_std"] = temp_out_image[4]


# raw_ct_image, raw_lb_image, ct_image, nm_image, suv_nm_image, lb_image = om.get_images("005")
