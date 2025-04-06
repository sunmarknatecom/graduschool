idx = "001"

ct_path, nm_path = get_paths(idx)
lb_path = "D:\\99gradu\\labels\\"+idx+"_nifti_label.nii"

ct_objs = open_CT(ct_path)
nm_obj = open_NM(nm_path)
lb_image = open_LB(lb_path)


skip_list = get_transform_var(ct_objs, nm_obj)["final_result"]

tr_ct_image = transform_ct_image(ct_objs, nm_obj)
tr_lb_image = transform_label(ct_objs, nm_obj, lb_image)
re_nm_image = realign_nm_image(nm_obj, skip_list)

nor_tr_ct_image = cv2.normalize(tr_ct_image, None, 0, 255, cv2.NORM_MINMAX)
nor_tr_lb_image = cv2.normalize(tr_lb_image, None, 0, 255, cv2.NORM_MINMAX)
nor_re_nm_image = cv2.normalize(re_nm_image, None, 0, 255, cv2.NORM_MINMAX)

nor_tr_ct_image = np.array(nor_tr_ct_image, dtype=np.uint8)
nor_tr_lb_image = np.array(nor_tr_lb_image, dtype=np.uint8)
nor_re_nm_image = np.array(nor_re_nm_image, dtype=np.uint8)

logs = '''Skull - Segment_91
Rt. humerus - Segment_70
Lt. humerus - Segment_69
Rt. femur - Segment_76
Lt. femur - Segment_75

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

color_nor_tr_ct_image = np.array([cv2.cvtColor(elem, cv2.COLOR_GRAY2RGB) for elem in nor_tr_ct_image], dtype=np.uint8)
color_nor_tr_lb_image = np.array([cv2.cvtColor(elem, cv2.COLOR_GRAY2RGB) for elem in nor_tr_lb_image], dtype=np.uint8)
color_nor_re_nm_image = np.array([cv2.cvtColor(elem, cv2.COLOR_GRAY2RGB) for elem in nor_re_nm_image], dtype=np.uint8)

def to_red_image(src_images):
    frame, width, height = np.shape(src_images)
    temp_ret_image = np.zeros((frame, width, height, 3), dtype=np.uint8)
    temp_ret_image[...,0] = src_images
    return temp_ret_image

red_color_nor_tr_lb_image = to_red_image(color_nor_tr_lb_image)

# 원소를 모두 255로

out_image = copy.copy(red_color_nor_tr_lb_image)
out_image[out_image>=1]=255


