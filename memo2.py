import orbismarci as om
import pandas as pd
import os
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


# long bones
long_bones = [69, 70, 75, 76]
out_df = []
for idx in idx_list[:30]:
    print("Starting to process", idx)
    raw_ct_image, raw_lb_image, ct_image, nm_image, suv_nm_image, lb_image = om.get_images(idx)
    print("Loaded images")
    volume = om.get_nm_vol_info(idx)
    temp_dict = {"idx":idx}
    for elem in long_bones:
        temp_lb_image = om.extract_binary_mask_label(lb_image, seg_n = int(elem))
        ranges = om.find_sig_lb_index_range(temp_lb_image)
        ranges = max(ranges, key=lambda x: x[1]-x[0])
        temp_lb_image[:ranges[0], :, :] = 0
        temp_lb_image[ranges[1]:, :, :] = 0
        temp_out_image = om.get_nm_stat_info(suv_nm_image, temp_lb_image)
        print(
            f"idx: {idx}, seg: {elem}, "
            f"volume: {volume * temp_out_image[0]:.2f}, "
            f"min: {temp_out_image[2]:.2f}, "
            f"max: {temp_out_image[1]:.2f}, "
            f"mean: {temp_out_image[3]:.2f}, "
            f"std: {temp_out_image[4]:.2f}"
        )
        temp_dict[organs[int(elem)]+"_vol"] = volume * temp_out_image[0]
        temp_dict[organs[int(elem)]+"_min"] = temp_out_image[2]
        temp_dict[organs[int(elem)]+"_max"] = temp_out_image[1]
        temp_dict[organs[int(elem)]+"_mean"] = temp_out_image[3]
        temp_dict[organs[int(elem)]+"_std"] = temp_out_image[4]
        out_df.append(temp_dict)
    print("Finished processing", idx)

long_bones_result_df = pd.DataFrame(out_df)
long_bones_result_df.to_csv("long_bones_result.csv", index=True)

# shpere bone
sphere_bones = [91]

# cube bones
cube_bones = [27, 28, 29]



# confirmation of ranges numbers
for idx in idx_list[:50]:
    _, __, ___, ____, ______, lb_image = om.get_images(idx)
    temp_lb_image = om.extract_binary_mask_label(lb_image, seg_n = 69)
    ranges = om.find_sig_lb_index_range(temp_lb_image)
    print(idx, ranges)
