idx = "001"

ct_path, nm_path = get_paths(idx)
lb_path = "D:\\99gradu\\labels\\"+idx+"\\_nifti_label.nii"

ct_objs = open_CT(ct_path)
nm_obj = open_NM(nm_path)
lb_image = open_LB(lb_path)


