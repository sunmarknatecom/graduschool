tr_lb_image = transform_label(ct_objs, nm_obj, lb_image)

for i in bones_index:
	temp_only_arr = only_seg_lb_1ch_image(lb_image, seg_n=i)
	temp_tr_lb_image = transform_label(ct_objs, nm_obj, temp_only_arr)
	temp_sig_frames = find_sig_frame(temp_only_arr)
	temp_index_list = find_sig_index(temp_sig_frames)
	print(organs[i], temp_index_list)

bones_index = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117]

organs = {
    1 : "spleen",
    2 : "kidney_right",
    3 : "kidney_left",
    4 : "gallbladder",
    5 : "liver",
    6 : "stomach",
    7 : "pancreas",
    8 : "adrenal_gland_right",
    9 : "adrenal_gland_left",
    10 : "lung_upper_lobe_left",
    11 : "lung_lower_lobe_left",
    12 : "lung_upper_lobe_right",
    13 : "lung_middle_lobe_right",
    14 : "lung_lower_lobe_right",
    15 : "esophagus",
    16 : "trachea",
    17 : "thyroid_gland",
    18 : "small_bowel",
    19 : "duodenum",
    20 : "colon",
    21 : "urinary_bladder",
    22 : "prostate",
    23 : "kidney_cyst_left",
    24 : "kidney_cyst_right",
    25 : "sacrum",
    26 : "vertebrae_S1",
    27 : "vertebrae_L5",
    28 : "vertebrae_L4",
    29 : "vertebrae_L3",
    30 : "vertebrae_L2",
    31 : "vertebrae_L1",
    32 : "vertebrae_T12",
    33 : "vertebrae_T11",
    34 : "vertebrae_T10",
    35 : "vertebrae_T9",
    36 : "vertebrae_T8",
    37 : "vertebrae_T7",
    38 : "vertebrae_T6",
    39 : "vertebrae_T5",
    40 : "vertebrae_T4",
    41 : "vertebrae_T3",
    42 : "vertebrae_T2",
    43 : "vertebrae_T1",
    44 : "vertebrae_C7",
    45 : "vertebrae_C6",
    46 : "vertebrae_C5",
    47 : "vertebrae_C4",
    48 : "vertebrae_C3",
    49 : "vertebrae_C2",
    50 : "vertebrae_C1",
    51 : "heart",
    52 : "aorta",
    53 : "pulmonary_vein",
    54 : "brachiocephalic_trunk",
    55 : "subclavian_artery_right",
    56 : "subclavian_artery_left",
    57 : "common_carotid_artery_right",
    58 : "common_carotid_artery_left",
    59 : "brachiocephalic_vein_left",
    60 : "brachiocephalic_vein_right",
    61 : "atrial_appendage_left",
    62 : "superior_vena_cava",
    63 : "inferior_vena_cava",
    64 : "portal_vein_and_splenic_vein",
    65 : "iliac_artery_left",
    66 : "iliac_artery_right",
    67 : "iliac_vena_left",
    68 : "iliac_vena_right",
    69 : "humerus_left",
    70 : "humerus_right",
    71 : "scapula_left",
    72 : "scapula_right",
    73 : "clavicula_left",
    74 : "clavicula_right",
    75 : "femur_left",
    76 : "femur_right",
    77 : "hip_left",
    78 : "hip_right",
    79 : "spinal_cord",
    80 : "gluteus_maximus_left",
    81 : "gluteus_maximus_right",
    82 : "gluteus_medius_left",
    83 : "gluteus_medius_right",
    84 : "gluteus_minimus_left",
    85 : "gluteus_minimus_right",
    86 : "autochthon_left",
    87 : "autochthon_right",
    88 : "iliopsoas_left",
    89 : "iliopsoas_right",
    90 : "brain",
    91 : "skull",
    92 : "rib_left_1",
    93 : "rib_left_2",
    94 : "rib_left_3",
    95 : "rib_left_4",
    96 : "rib_left_5",
    97 : "rib_left_6",
    98 : "rib_left_7",
    99 : "rib_left_8",
    100 : "rib_left_9",
    101 : "rib_left_10",
    102 : "rib_left_11",
    103 : "rib_left_12",
    104 : "rib_right_1",
    105 : "rib_right_2",
    106 : "rib_right_3",
    107 : "rib_right_4",
    108 : "rib_right_5",
    109 : "rib_right_6",
    110 : "rib_right_7",
    111 : "rib_right_8",
    112 : "rib_right_9",
    113 : "rib_right_10",
    114 : "rib_right_11",
    115 : "rib_right_12",
    116 : "sternum",
    117 : "costal_cartilages"
}

# main 함수

idx = ".\\data\\001"

ct_path, nm_path = get_paths(idx)

ct_objs = open_CT(ct_path)
nm_obj = open_NM(nm_path)
lb_path = "D:\\gradustudy\\labels\\"+idx[-3:]+"_nifti_label.nii"

ct_image = create_ct_image(ct_objs)
nm_image = nm_obj.pixel_array
lb_image = open_LB(lb_path)

skip_list = get_transform_var(ct_objs, nm_obj)["final result"]
re_nm_image = realign_nm_image(nm_obj, skip_list)

organ_index_dict = {}

for idx, i in enumerate(bones_index):
    temp_subset_lb_image = only_seg_lb_1ch_image(lb_image, seg_n=i)
    temp_tr_subset_lb_image = transform_label(ct_objs, nm_obj, temp_subset_lb_image)
    temp_sig_frames = find_sig_frame(temp_tr_subset_lb_image)
    temp_index_list = find_sig_index(temp_sig_frames)
    organ_index_dict[organs[i]]=temp_index_list
    print(organs[i], temp_index_list)


def main():
    bones_index = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117]
    root_idx = [".\\data\\001",".\\data\\002",".\\data\\003"]
    for IDX in root_idx:
        ct_path, nm_path = get_paths(IDX)
        ct_objs = open_CT(ct_path)
        nm_obj = open_NM(nm_path)
        lb_path = "D:\\gradustudy\\labels\\"+IDX[-3:]+"_nifti_label.nii"
        ct_image = create_ct_image(ct_objs)
        nm_image = nm_obj.pixel_array
        lb_image = open_LB(lb_path)
        skip_list = get_transform_var(ct_objs, nm_obj)["final result"]
        re_nm_image = realign_nm_image(nm_obj, skip_list)
        organ_index_dict = {}
        for i in bones_index:
            temp_subset_lb_image = only_seg_lb_1ch_image(lb_image, seg_n=i)
            temp_tr_subset_lb_image = transform_label(ct_objs, nm_obj, temp_subset_lb_image)
            temp_sig_frames = find_sig_frame(temp_tr_subset_lb_image)
            temp_index_list = find_sig_index(temp_sig_frames)
            organ_index_dict[organs[i]]=temp_index_list
            print(organs[i], temp_index_list)

def get_ct_nm_lb_images(idx):
    temp_ct_path, temp_nm_path = get_paths(".\\data\\"+idx)
    temp_lb_path = "D:\\gradustudy\\labels\\"+idx+"_nifti_label.nii"
    temp_ct_objs = open_CT(temp_ct_path)
    temp_nm_obj = open_NM(temp_nm_path)
    temp_lb_image = open_LB(temp_lb_path)
    tr_temp_ct_image = transform_ct_image(temp_ct_objs, temp_nm_obj)
    tr_temp_lb_image = transform_label(temp_ct_objs, temp_nm_obj, temp_lb_image)
    temp_skip_list = get_transform_var(temp_ct_objs, temp_nm_obj)["final result"]
    re_nm_image = realign_nm_image(nm_obj, temp_skip_list)
    return tr_temp_ct_image, re_nm_image, tr_temp_lb_image

for elem in idx_list:
    ct_iamge, nm_image, lb_image = get_ct_nm_lb_images(elem)
    print(elem, np.shape(ct_iamge), np.shape(nm_image), np.shape(lb_image))
