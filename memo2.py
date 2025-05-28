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




# long bones
import orbismarci as om
import pandas as pd
import os, copy
import numpy as np
import time
import datetime

def long_bone_analyze(result_file_name = "long_bones_result_all.csv"):
    idx_list = os.listdir(".\\data\\")
    bone_index = om.get_bone_indices()
    organs = om.get_organs()
    long_bones = [69, 70, 75, 76]
    out_df = []
    for idx in idx_list:
        start_time = time.time()
        start_datetime = datetime.datetime.now()
        print("Starting to process", idx)
        # time Record
        print(f"Start Time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        # time Record
        raw_ct_image, raw_lb_image, ct_image, nm_image, suv_nm_image, lb_image = om.get_images(idx)
        print("Loaded images")
        volume = om.get_nm_vol_info(idx)
        temp_dict = {"idx":idx}
        for elem in long_bones:
            temp_lb_image = om.extract_binary_mask_label(lb_image, seg_n = int(elem))
            ranges = om.find_sig_lb_index_range(temp_lb_image)
            ranges = max(ranges, key=lambda x: x[1]-x[0])
            center = int((ranges[0]+ranges[1])/2)
            temp_1fr_lb_image = copy.copy(temp_lb_image)
            temp_3fr_lb_image = copy.copy(temp_lb_image)
            temp_5fr_lb_image = copy.copy(temp_lb_image)
            # 1 slice
            temp_1fr_lb_image[:center, :, :] = 0
            temp_1fr_lb_image[center+1:, :, :] = 0
            # 3 slice
            temp_3fr_lb_image[:center-1, :, :] = 0
            temp_3fr_lb_image[center+2:, :, :] = 0
            # 5 slice
            temp_5fr_lb_image[:center-2, :, :] = 0
            temp_5fr_lb_image[center+3:, :, :] = 0
            # results
            temp_out_1fr_image = om.get_nm_stat_info(suv_nm_image, temp_1fr_lb_image)
            temp_out_3fr_image = om.get_nm_stat_info(suv_nm_image, temp_3fr_lb_image)
            temp_out_5fr_image = om.get_nm_stat_info(suv_nm_image, temp_5fr_lb_image)
            print(
                f"idx: {idx}, seg: {elem}, "
                f"volume: {volume * temp_out_1fr_image[0]:.2f}, "
                f"min: {temp_out_1fr_image[2]:.2f}, "
                f"max: {temp_out_1fr_image[1]:.2f}, "
                f"mean: {temp_out_1fr_image[3]:.2f}, "
                f"std: {temp_out_1fr_image[4]:.2f}"
                f"3_fr_volume: {volume * temp_out_3fr_image[0]:.2f}, "
                f"3_fr_min: {temp_out_3fr_image[2]:.2f}, "
                f"3_fr_max: {temp_out_3fr_image[1]:.2f}, "
                f"3_fr_mean: {temp_out_3fr_image[3]:.2f}, "
                f"3_fr_std: {temp_out_3fr_image[4]:.2f}"
                f"5_fr_volume: {volume * temp_out_5fr_image[0]:.2f}, "
                f"5_fr_min: {temp_out_5fr_image[2]:.2f}, "
                f"5_fr_max: {temp_out_5fr_image[1]:.2f}, "
                f"5_fr_mean: {temp_out_5fr_image[3]:.2f}, "
                f"5_fr_std: {temp_out_5fr_image[4]:.2f}"
            )
            temp_dict[organs[int(elem)]+"_range"] = ranges
            temp_dict[organs[int(elem)]+"_center_slice"] = center
            temp_dict[organs[int(elem)]+"_vol"] = volume * temp_out_1fr_image[0]
            temp_dict[organs[int(elem)]+"_min"] = temp_out_1fr_image[2]
            temp_dict[organs[int(elem)]+"_max"] = temp_out_1fr_image[1]
            temp_dict[organs[int(elem)]+"_mean"] = temp_out_1fr_image[3]
            temp_dict[organs[int(elem)]+"_std"] = temp_out_1fr_image[4]
            # 3frame
            temp_dict[organs[int(elem)]+"_3fr_vol"] = volume * temp_out_3fr_image[0]
            temp_dict[organs[int(elem)]+"_3fr_min"] = temp_out_3fr_image[2]
            temp_dict[organs[int(elem)]+"_3fr_max"] = temp_out_3fr_image[1]
            temp_dict[organs[int(elem)]+"_3fr_mean"] = temp_out_3fr_image[3]
            temp_dict[organs[int(elem)]+"_3fr_std"] = temp_out_3fr_image[4]
            # 5frame
            temp_dict[organs[int(elem)]+"_5fr_vol"] = volume * temp_out_5fr_image[0]
            temp_dict[organs[int(elem)]+"_5fr_min"] = temp_out_5fr_image[2]
            temp_dict[organs[int(elem)]+"_5fr_max"] = temp_out_5fr_image[1]
            temp_dict[organs[int(elem)]+"_5fr_mean"] = temp_out_5fr_image[3]
            temp_dict[organs[int(elem)]+"_5fr_std"] = temp_out_5fr_image[4]
        out_df.append(temp_dict)
        # time Record
        end_time = time.time()
        end_datetime = datetime.datetime.now()
        print(f"End Time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        elapsed_time_seconds = end_time - start_time
        elapsed_timedelta = end_datetime - start_datetime
        # time Record
        print("Finished processing", idx)
    long_bones_result_df = pd.DataFrame(out_df)
    long_bones_result_df.to_csv(result_file_name, index=True)

# shpere bone

def sphere_bone_analyze(result_file_name = "sphere_bones_result_all.csv"):
    idx_list = os.listdir(".\\data\\")
    organs = om.get_organs()
    sphere_bones = [91]
    out_df = []
    for idx in idx_list[:5]:
        start_time = time.time()
        start_datetime = datetime.datetime.now()
        print("Starting to process", idx)
        # time Record
        print(f"Start Time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        # time Record
        raw_ct_image, raw_lb_image, ct_image, nm_image, suv_nm_image, lb_image = om.get_images(idx)
        print("Loaded images")
        volume = om.get_nm_vol_info(idx)
        temp_dict = {"idx":idx}
        for elem in sphere_bones:
            temp_lb_image = om.extract_binary_mask_label(lb_image, seg_n = int(elem))
            ranges = om.find_sig_lb_index_range(temp_lb_image)
            ranges = max(ranges, key=lambda x: x[1]-x[0])
            center = int((ranges[0]+ranges[1]*2)/3)
            temp_rearr_lb_image = copy.copy(temp_lb_image)
            temp_rearr_lb_image[:center-2, :, :] = 0
            temp_rearr_lb_image[center+3:, :, :] = 0
            ranges2 = om.find_axial_sig_lb_index_range(temp_rearr_lb_image, axial_index=1)
            ranges2 = max(ranges2, key=lambda x: x[1]-x[0])
            center2 = int((ranges2[0]+ranges2[1])/2)
            temp_rearr_lb_image[:,:center2-2,:] = 0
            temp_rearr_lb_image[:,center2+3:,:] = 0
            ranges3 = om.find_axial_sig_lb_index_range(temp_rearr_lb_image, axial_index=2)
            rt_range = ranges3[0]
            lt_range = ranges3[1]
            rt_temp_rearr_lb_image = copy.copy(temp_rearr_lb_image)
            lt_temp_rearr_lb_image = copy.copy(temp_lb_image)
            rt_temp_rearr_lb_image[:,:,:rt_range[0]] = 0
            rt_temp_rearr_lb_image[:,:,rt_range[1]:] = 0
            lt_temp_rearr_lb_image[:,:,:lt_range[0]] = 0
            lt_temp_rearr_lb_image[:,:,lt_range[1]:] = 0
            # results
            rt_temp_out_report = om.get_nm_stat_info(suv_nm_image, rt_temp_rearr_lb_image)
            lt_temp_out_report = om.get_nm_stat_info(suv_nm_image, lt_temp_rearr_lb_image)
            print(
                f"idx: {idx}, seg: {elem}, "
                f"rt_skull volume: {volume * rt_temp_out_report[0]:.2f}, "
                f"rt_min: {rt_temp_out_report[2]:.2f}, "
                f"rt_max: {rt_temp_out_report[1]:.2f}, "
                f"rt_mean: {rt_temp_out_report[3]:.2f}, "
                f"rt_std: {rt_temp_out_report[4]:.2f}"
                f"lt_skull volume: {volume * lt_temp_out_report[0]:.2f}, "
                f"lt_min: {lt_temp_out_report[2]:.2f}, "
                f"lt_max: {lt_temp_out_report[1]:.2f}, "
                f"lt_mean: {lt_temp_out_report[3]:.2f}, "
                f"lt_std: {lt_temp_out_report[4]:.2f}"
            )
            temp_dict[organs[int(elem)]+"_rt_range"] = ranges
            temp_dict[organs[int(elem)]+"_rt_center_slice"] = center
            temp_dict[organs[int(elem)]]
            #
            # 작성중
            #
            temp_dict[organs[int(elem)]+"_vol"] = volume * temp_out_image[0]
            temp_dict[organs[int(elem)]+"_min"] = temp_out_image[2]
            temp_dict[organs[int(elem)]+"_max"] = temp_out_image[1]
            temp_dict[organs[int(elem)]+"_mean"] = temp_out_image[3]
            temp_dict[organs[int(elem)]+"_std"] = temp_out_image[4]
        out_df.append(temp_dict)
        # time Record
        end_time = time.time()
        end_datetime = datetime.datetime.now()
        print(f"End Time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        elapsed_time_seconds = end_time - start_time
        elapsed_timedelta = end_datetime - start_datetime
        # time Record
        print("Finished processing", idx)
        print(f"Elapsed time: {elapsed_time_seconds:.2f} seconds")
    long_bones_result_df = pd.DataFrame(out_df)
    long_bones_result_df.to_csv(result_file_name, index=True)

# cube bones


def cube_bone_analyze(result_file_name = "sphere_bones_result_all.csv"):
    idx_list = os.listdir(".\\data\\")
    organs = om.get_organs()
    cube_bones = [27, 28, 29]
    out_df = []
    for idx in idx_list:
        start_time = time.time()
        start_datetime = datetime.datetime.now()
        print("Starting to process", idx)
        # time Record
        print(f"Start Time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        # time Record
        raw_ct_image, raw_lb_image, ct_image, nm_image, suv_nm_image, lb_image = om.get_images(idx)
        print("Loaded images")
        volume = om.get_nm_vol_info(idx)
        temp_dict = {"idx":idx}
        for elem in cube_bones:
            temp_lb_image = om.extract_binary_mask_label(lb_image, seg_n = int(elem))
            ranges = om.find_sig_lb_index_range(temp_lb_image)
            ranges = max(ranges, key=lambda x: x[1]-x[0])
            center = int((ranges[0]+ranges[1])/2)
            temp_1fr_lb_image = copy.copy(temp_lb_image)
            temp_3fr_lb_image = copy.copy(temp_lb_image)
            temp_5fr_lb_image = copy.copy(temp_lb_image)
            # 1 slice
            temp_1fr_lb_image[:center, :, :] = 0
            temp_1fr_lb_image[center+1:, :, :] = 0
            # 3 slice
            temp_3fr_lb_image[:center-1, :, :] = 0
            temp_3fr_lb_image[center+2:, :, :] = 0
            # 5 slice
            temp_5fr_lb_image[:center-2, :, :] = 0
            temp_5fr_lb_image[center+3:, :, :] = 0
            # results
            temp_out_1fr_image = om.get_nm_stat_info(suv_nm_image, temp_1fr_lb_image)
            temp_out_3fr_image = om.get_nm_stat_info(suv_nm_image, temp_3fr_lb_image)
            temp_out_5fr_image = om.get_nm_stat_info(suv_nm_image, temp_5fr_lb_image)
            print(
                f"idx: {idx}, seg: {elem}, "
                f"volume: {volume * temp_out_1fr_image[0]:.2f}, "
                f"min: {temp_out_1fr_image[2]:.2f}, "
                f"max: {temp_out_1fr_image[1]:.2f}, "
                f"mean: {temp_out_1fr_image[3]:.2f}, "
                f"std: {temp_out_1fr_image[4]:.2f}"
                f"3_fr_volume: {volume * temp_out_3fr_image[0]:.2f}, "
                f"3_fr_min: {temp_out_3fr_image[2]:.2f}, "
                f"3_fr_max: {temp_out_3fr_image[1]:.2f}, "
                f"3_fr_mean: {temp_out_3fr_image[3]:.2f}, "
                f"3_fr_std: {temp_out_3fr_image[4]:.2f}"
                f"5_fr_volume: {volume * temp_out_5fr_image[0]:.2f}, "
                f"5_fr_min: {temp_out_5fr_image[2]:.2f}, "
                f"5_fr_max: {temp_out_5fr_image[1]:.2f}, "
                f"5_fr_mean: {temp_out_5fr_image[3]:.2f}, "
                f"5_fr_std: {temp_out_5fr_image[4]:.2f}"
            )
            temp_dict[organs[int(elem)]+"_range"] = ranges
            temp_dict[organs[int(elem)]+"_center_slice"] = center
            temp_dict[organs[int(elem)]+"_vol"] = volume * temp_out_1fr_image[0]
            temp_dict[organs[int(elem)]+"_min"] = temp_out_1fr_image[2]
            temp_dict[organs[int(elem)]+"_max"] = temp_out_1fr_image[1]
            temp_dict[organs[int(elem)]+"_mean"] = temp_out_1fr_image[3]
            temp_dict[organs[int(elem)]+"_std"] = temp_out_1fr_image[4]
            # 3frame
            temp_dict[organs[int(elem)]+"_3fr_vol"] = volume * temp_out_3fr_image[0]
            temp_dict[organs[int(elem)]+"_3fr_min"] = temp_out_3fr_image[2]
            temp_dict[organs[int(elem)]+"_3fr_max"] = temp_out_3fr_image[1]
            temp_dict[organs[int(elem)]+"_3fr_mean"] = temp_out_3fr_image[3]
            temp_dict[organs[int(elem)]+"_3fr_std"] = temp_out_3fr_image[4]
            # 5frame
            temp_dict[organs[int(elem)]+"_5fr_vol"] = volume * temp_out_5fr_image[0]
            temp_dict[organs[int(elem)]+"_5fr_min"] = temp_out_5fr_image[2]
            temp_dict[organs[int(elem)]+"_5fr_max"] = temp_out_5fr_image[1]
            temp_dict[organs[int(elem)]+"_5fr_mean"] = temp_out_5fr_image[3]
            temp_dict[organs[int(elem)]+"_5fr_std"] = temp_out_5fr_image[4]
        out_df.append(temp_dict)
        # time Record
        end_time = time.time()
        end_datetime = datetime.datetime.now()
        print(f"End Time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        elapsed_time_seconds = end_time - start_time
        elapsed_timedelta = end_datetime - start_datetime
        # time Record
        print("Finished processing", idx)
    long_bones_result_df = pd.DataFrame(out_df)
    long_bones_result_df.to_csv(result_file_name, index=True)
