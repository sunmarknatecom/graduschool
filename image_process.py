import segspect as sgs

def to_red_image(src_images):
    temp_image = np.array([cv2.normalize(elem, None, 0, 255, cv2.NORM_MINMAX) for elem in src_images],dtype=np.uint8)
    temp_image = np.array([cv2.cvtColor(elem, cv2.COLOR_GRAY2RGB)],dtype=np.uint8)
    temp_image[:,:,:,2]=0
    temp_image[:,:,:,1]=0
    return temp_image

def to_color_image(src_images):
    #normalize
    norm_images = np.array([cv2.normalize(elem, None, 0, 255, cv2.NORM_MINMAX) for elem in src_images],dtype=np.uint8)
    return np.array([cv2.cvtColor(elem, cv2.COLOR_GRAY2RGB) for elem in norm_images], dtype=np.uint8)

def only_seg_lb_image(src_lb_image, seg_n = 70):
    return (src_lb_image == seg_n).astype(np.uint8)*seg_n

def only_seg_lb_1ch_image(src_lb_image, seg_n = 70):
    return (src_lb_image == seg_n).astype(np.uint8)

def find_min_max_index(src_lb_image, seg_n = 70):
    indices = np.argwhere(src_lb_image == seg_n)
    return np.min(indices[:,0]), np.max(indices[:,0])

def find_sig_index(arr):
    ranges = []
    start = None
    for i, elem in enumerate(arr):
        if elem != 0:
            if start is None:
                start = i
        else:
            if start is not None:
                ranges.append((start, i))
                start = None
    if start is not None:
        ranges.append((start, len(arr)))
    return ranges

def find_sig_frame(arr):
    return [(np.sum(elem) !=0).astype(int) for elem in arr]

idx = "001"

ct_path, nm_path = sgs.get_paths(idx)
lb_path = "D:\\99gradu\\labels\\"+idx+"_nifti_label.nii"

ct_objs = sgs.open_CT(ct_path)
nm_obj = sgs.open_NM(nm_path)
lb_image = sgs.open_LB(lb_path)


skip_list = sgs.get_transform_var(ct_objs, nm_obj)["final result"]

tr_ct_image = sgs.transform_ct_image(ct_objs, nm_obj)
tr_lb_image = sgs.transform_label(ct_objs, nm_obj, lb_image)
re_nm_image = sgs.realign_nm_image(nm_obj, skip_list)

# normalization of images

nor_tr_ct_image = cv2.normalize(tr_ct_image, None, 0, 255, cv2.NORM_MINMAX)
nor_tr_lb_image = cv2.normalize(tr_lb_image, None, 0, 255, cv2.NORM_MINMAX)
nor_re_nm_image = cv2.normalize(re_nm_image, None, 0, 255, cv2.NORM_MINMAX)

# dtype to uint8

nor_tr_ct_image = np.array(nor_tr_ct_image, dtype=np.uint8)
nor_tr_lb_image = np.array(nor_tr_lb_image, dtype=np.uint8)
nor_re_nm_image = np.array(nor_re_nm_image, dtype=np.uint8)

logs = '''Skull - Segment_91
Rt. humerus - Segment_70
Lt. humerus - Segment_69
Rt. femur - Segment_76
Lt. femur - Segment_75
skull - Segment_91
sternum - Segment_116
Rt. pelvis - Segment_78
Lt. pelvis - Segment_77

C1 - Segment_50
C2 - Segment_49
C3 - Segment_48
C4 - Segment_47
C5 - Segment_46
C6 - Segment_45
C7 - Segment_44

T1 - Segment_43
T2 - Segment_42
T3 - Segment_41
T4 - Segment_40
T5 - Segment_39
T6 - Segment_38
T7 - Segment_37
T8 - Segment_36
T9 - Segment_35
T10 - Segment_34
T11 - Segment_33
T12 - Segment_32

L1 - Segment_31
L2 - Segment_30
L3 - Segment_29
L4 - Segment_28
L5 - Segment_27
'''

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

params = {"C1 Volume": volC1,
    "C1 maxSUV": maxSUVC1,
    "C1 meanSUV": meanSUVC1,
    "C1 stdSUV": stdSUVC1,
    "C2 Volume": volC2,
    "C2 maxSUV": maxSUVC2,
    "C2 meanSUV": meanSUVC2,
    "C2 stdSUV": stdSUVC2,
    "C3 Volume": volC3,
    "C3 maxSUV": maxSUVC3,
    "C3 meanSUV": meanSUVC3,
    "C3 stdSUV": stdSUVC3,
    "C4 Volume": volC4,
    "C4 maxSUV": maxSUVC4,
    "C4 meanSUV": meanSUVC4,
    "C4 stdSUV": stdSUVC4,
    "C5 Volume": volC5,
    "C5 maxSUV": maxSUVC5,
    "C5 meanSUV": meanSUVC5,
    "C5 stdSUV": stdSUVC5,
    "C6 Volume": volC6,
    "C6 maxSUV": maxSUVC6,
    "C6 meanSUV": meanSUVC6,
    "C6 stdSUV": stdSUVC6,
    "C7 Volume": volC7,
    "C7 maxSUV": maxSUVC7,
    "C7 meanSUV": meanSUVC7,
    "C7 stdSUV": stdSUVC7,
    "T1 Volume": volT1,
    "T1 maxSUV": maxSUVT1,
    "T1 meanSUV": meanSUVT1,
    "T1 stdSUV": stdSUVT1,
    "T2 Volume": volT2,
    "T2 maxSUV": maxSUVT2,
    "T2 meanSUV": meanSUVT2,
    "T2 stdSUV": stdSUVT2,
    "T3 Volume": volT3,
    "T3 maxSUV": maxSUVT3,
    "T3 meanSUV": meanSUVT3,
    "T3 stdSUV": stdSUVT3,
    "T4 Volume": volT4,
    "T4 maxSUV": maxSUVT4,
    "T4 meanSUV": meanSUVT4,
    "T4 stdSUV": stdSUVT4,
    "T5 Volume": volT5,
    "T5 maxSUV": maxSUVT5,
    "T5 meanSUV": meanSUVT5,
    "T5 stdSUV": stdSUVT5,
    "T6 Volume": volT6,
    "T6 maxSUV": maxSUVT6,
    "T6 meanSUV": meanSUVT6,
    "T6 stdSUV": stdSUVT6,
    "T7 Volume": volT7,
    "T7 maxSUV": maxSUVT7,
    "T7 meanSUV": meanSUVT7,
    "T7 stdSUV": stdSUVT7,
    "T8 Volume": volT8,
    "T8 maxSUV": maxSUVT8,
    "T8 meanSUV": meanSUVT8,
    "T8 stdSUV": stdSUVT8,
    "T9 Volume": volT9,
    "T9 maxSUV": maxSUVT9,
    "T9 meanSUV": meanSUVT9,
    "T9 stdSUV": stdSUVT9,
    "T10 Volume": volT10,
    "T10 maxSUV": maxSUVT10,
    "T10 meanSUV": meanSUVT10,
    "T10 stdSUV": stdSUVT10,
    "T11 Volume": volT11,
    "T11 maxSUV": maxSUVT11,
    "T11 meanSUV": meanSUVT11,
    "T11 stdSUV": stdSUVT11,
    "T12 Volume": volT12,
    "T12 maxSUV": maxSUVT12,
    "T12 meanSUV": meanSUVT12,
    "T12 stdSUV": stdSUVT12,
    "L1 Volume": volL1,
    "L1 maxSUV": maxSUVL1,
    "L1 meanSUV": meanSUVL1,
    "L1 stdSUV": stdSUVL1,
    "L2 Volume": volL2,
    "L2 maxSUV": maxSUVL2,
    "L2 meanSUV": meanSUVL2,
    "L2 stdSUV": stdSUVL2,
    "L3 Volume": volL3,
    "L3 maxSUV": maxSUVL3,
    "L3 meanSUV": meanSUVL3,
    "L3 stdSUV": stdSUVL3,
    "L4 Volume": volL4,
    "L4 maxSUV": maxSUVL4,
    "L4 meanSUV": meanSUVL4,
    "L4 stdSUV": stdSUVL4,
    "L5 Volume": volL5,
    "L5 maxSUV": maxSUVL5,
    "L5 meanSUV": meanSUVL5,
    "L5 stdSUV": stdSUVL5,
    "sacrum Volume": volSacrum,
    "sacrum maxSUV": maxSUVSacrum,
    "sacrum meanSUV": meanSUVSacrum,
    "sacrum stdSUV": stdSUVSacrum,
    "Rt. humerus Volume": volRtHumerus,
    "Rt. humerus maxSUV": maxSUVRtHumerus,
    "Rt. humerus meanSUV": meanSUVRtHumerus,
    "Rt. humerus stdSUV": stdSUVRtHumerus,
    "Lt. humerus Volume": volLtHumerus,
    "Lt. humerus maxSUV": maxSUVLtHumerus,
    "Lt. humerus meanSUV": meanSUVLtHumerus,
    "Lt. humerus stdSUV": stdSUVLtHumerus,
    "Rt. femur Volume": volRtFemur,
    "Rt. femur maxSUV": maxSUVRtFemur,
    "Rt. femur meanSUV": meanSUVRtFemur,
    "Rt. femur stdSUV": stdSUVRtFemur,
    "Lt. femur Volume": volLtFemur,
    "Lt. femur maxSUV": maxSUVLtFemur,
    "Lt. femur meanSUV": meanSUVLtFemur,
    "Lt. femur stdSUV": stdSUVLtFemur,
    "skull Volume": volSkull,
    "skull maxSUV": maxSUVSkull,
    "skull meanSUV": meanSUVSkull,
    "skull stdSUV": stdSUVSkull,
    "Rt. pelvis Volume" : volRtPelvis,
    "Rt. pelvis maxSUV" : maxSUVRtPelvis,
    "Rt. pelvis meanSUV" : meanSUVRtPelvis,
    "Rt. pelvis stdSUV" : stdSUVRtPelvis,
    "Lt. pelvis Volume" : volLtPelvis,
    "Lt. pelvis maxSUV" : maxSUVLtPelvis,
    "Lt. pelvis meanSUV" : meanSUVLtPelvis,
    "Lt. pelvis stdSUV" : stdSUVLtPelvis,
    "sternum Volume" : volSternum,
    "sternum maxSUV" : maxSUVSternum,
    "sternum meanSUV" : meanSUVSternum,
    "sternum stdSUV" : stdSUVSternum    
}

# gray to color

color_nor_tr_ct_image = np.array([cv2.cvtColor(elem, cv2.COLOR_GRAY2RGB) for elem in nor_tr_ct_image], dtype=np.uint8)
color_nor_tr_lb_image = np.array([cv2.cvtColor(elem, cv2.COLOR_GRAY2RGB) for elem in nor_tr_lb_image], dtype=np.uint8)
color_nor_re_nm_image = np.array([cv2.cvtColor(elem, cv2.COLOR_GRAY2RGB) for elem in nor_re_nm_image], dtype=np.uint8)

red_color_nor_tr_lb_image = sgs.to_red_image(color_nor_tr_lb_image)

# 원소를 모두 255로
out_image = copy.copy(red_color_nor_tr_lb_image)
out_image[out_image>=1]=255
