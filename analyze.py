import orbismarci as om
import pandas as pd
import os, copy
import numpy as np
import time
import datetime
from scipy.ndimage import center_of_mass

# long bones

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
            distal_1thrid_frame = int(ranges[0] + (ranges[1]-ranges[0])*0.25)
            middle_frame = int(ranges[0] + (ranges[1]-ranges[0])*0.5)
            proxim_3third_frame = int(ranges[0] + (ranges[1]-ranges[0])*0.75)
            zdc = center_of_mass(temp_lb_image[distal_1thrid_frame])
            zmc = center_of_mass(temp_lb_image[middle_frame])
            zpc = center_of_mass(temp_lb_image[proxim_3third_frame])
            offset = 1
            volume_zdc = suv_nm_image[distal_c-offset:distal_c+offset+1, zdc[1]-offset:zdc[1]+offset+1, zdc[2]-offset:zdc[2]+offset+1]
            volume_q2c = suv_nm_image[zmc[0]-offset:zmc[0]+offset+1, zmc[1]-offset:zmc[1]+offset+1, zmc[2]-offset:zmc[2]+offset+1]
            volume_q3c = suv_nm_image[zpc[0]-offset:zpc[0]+offset+1, zpc[1]-offset:zpc[1]+offset+1, zpc[2]-offset:zpc[2]+offset+1]
            temp_out_zdc = [np.shape(zdc)[0]*np.shape(zdc)[1]*np.shape(zdc)[2], np.max(volume_zdc), np.min(volume_zdc), np.mean(volume_zdc), np.std(volume_zdc)]
            temp_out_zmc = [np.shape(zmc)[0]*np.shape(zmc)[1]*np.shape(zmc)[2], np.max(volume_q2c), np.min(volume_q2c), np.mean(volume_q2c), np.std(volume_q2c)]
            temp_out_zpc = [np.shape(zpc)[0]*np.shape(zpc)[1]*np.shape(zpc)[2], np.max(volume_q3c), np.min(volume_q3c), np.mean(volume_q3c), np.std(volume_q3c)]
            # results

            print(
                f"idx: {idx}, seg: {elem}, "
                f"distal_volume: {volume * temp_out_zdc[0]:.2f}, "
                f"distal_min:    {temp_out_zdc[2]:.2f}, "
                f"distal_max:    {temp_out_zdc[1]:.2f}, "
                f"distal_mean:   {temp_out_zdc[3]:.2f}, "
                f"distal_std:    {temp_out_zdc[4]:.2f}"
                f"middle_volume: {volume * temp_out_zmc[0]:.2f}, "
                f"middle_min:    {temp_out_zmc[2]:.2f}, "
                f"middle_max:    {temp_out_zmc[1]:.2f}, "
                f"middle_mean:   {temp_out_zmc[3]:.2f}, "
                f"middle_std:    {temp_out_zmc[4]:.2f}"
                f"proxim_volume: {volume * temp_out_zpc[0]:.2f}, "
                f"proxim_min:    {temp_out_zpc[2]:.2f}, "
                f"proxim_max:    {temp_out_zpc[1]:.2f}, "
                f"proxim_mean:   {temp_out_zpc[3]:.2f}, "
                f"proxim_std:    {temp_out_zpc[4]:.2f}"
            )
            temp_dict[organs[int(elem)]+"_range"] = ranges
            temp_dict[organs[int(elem)]+"_centers"] = center
            frame_results = {
                "1fr": temp_out_1fr_image,
                "3fr": temp_out_3fr_image,
                "5fr": temp_out_5fr_image
            }
            for fr_label, stats in frame_results.items():
                temp_dict[organs[int(elem)] + f"_{fr_label}_vol"] = volume * stats[0]
                temp_dict[organs[int(elem)] + f"_{fr_label}_min"] = stats[2]
                temp_dict[organs[int(elem)] + f"_{fr_label}_max"] = stats[1]
                temp_dict[organs[int(elem)] + f"_{fr_label}_mean"] = stats[3]
                temp_dict[organs[int(elem)] + f"_{fr_label}_std"] =  stats[4]
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


# def long_bone_analyze(result_file_name = "long_bones_result_all.csv"):
#     idx_list = os.listdir(".\\data\\")
#     bone_index = om.get_bone_indices()
#     organs = om.get_organs()
#     long_bones = [69, 70, 75, 76]
#     out_df = []
#     for idx in idx_list:
#         start_time = time.time()
#         start_datetime = datetime.datetime.now()
#         print("Starting to process", idx)
#         # time Record
#         print(f"Start Time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
#         # time Record
#         raw_ct_image, raw_lb_image, ct_image, nm_image, suv_nm_image, lb_image = om.get_images(idx)
#         print("Loaded images")
#         volume = om.get_nm_vol_info(idx)
#         temp_dict = {"idx":idx}
#         for elem in long_bones:
#             temp_lb_image = om.extract_binary_mask_label(lb_image, seg_n = int(elem))
#             ranges = om.find_sig_lb_index_range(temp_lb_image)
#             ranges = max(ranges, key=lambda x: x[1]-x[0])
#             center = int((ranges[0]+ranges[1])/2)
#             temp_1fr_lb_image = copy.copy(temp_lb_image)
#             temp_3fr_lb_image = copy.copy(temp_lb_image)
#             temp_5fr_lb_image = copy.copy(temp_lb_image)
#             # 1 slice
#             temp_1fr_lb_image[:center, :, :] = 0
#             temp_1fr_lb_image[center+1:, :, :] = 0
#             # 3 slice
#             temp_3fr_lb_image[:center-1, :, :] = 0
#             temp_3fr_lb_image[center+2:, :, :] = 0
#             # 5 slice
#             temp_5fr_lb_image[:center-2, :, :] = 0
#             temp_5fr_lb_image[center+3:, :, :] = 0
#             # results
#             temp_out_1fr_image = om.get_nm_stat_info(suv_nm_image, temp_1fr_lb_image)
#             temp_out_3fr_image = om.get_nm_stat_info(suv_nm_image, temp_3fr_lb_image)
#             temp_out_5fr_image = om.get_nm_stat_info(suv_nm_image, temp_5fr_lb_image)
#             print(
#                 f"idx: {idx}, seg: {elem}, "
#                 f"volume: {volume * temp_out_1fr_image[0]:.2f}, "
#                 f"min: {temp_out_1fr_image[2]:.2f}, "
#                 f"max: {temp_out_1fr_image[1]:.2f}, "
#                 f"mean: {temp_out_1fr_image[3]:.2f}, "
#                 f"std: {temp_out_1fr_image[4]:.2f}"
#                 f"3_fr_volume: {volume * temp_out_3fr_image[0]:.2f}, "
#                 f"3_fr_min: {temp_out_3fr_image[2]:.2f}, "
#                 f"3_fr_max: {temp_out_3fr_image[1]:.2f}, "
#                 f"3_fr_mean: {temp_out_3fr_image[3]:.2f}, "
#                 f"3_fr_std: {temp_out_3fr_image[4]:.2f}"
#                 f"5_fr_volume: {volume * temp_out_5fr_image[0]:.2f}, "
#                 f"5_fr_min: {temp_out_5fr_image[2]:.2f}, "
#                 f"5_fr_max: {temp_out_5fr_image[1]:.2f}, "
#                 f"5_fr_mean: {temp_out_5fr_image[3]:.2f}, "
#                 f"5_fr_std: {temp_out_5fr_image[4]:.2f}"
#             )
#             temp_dict[organs[int(elem)]+"_range"] = ranges
#             temp_dict[organs[int(elem)]+"_center_slice"] = center
#             frame_results = {
#                 "1fr": temp_out_1fr_image,
#                 "3fr": temp_out_3fr_image,
#                 "5fr": temp_out_5fr_image
#             }
#             for fr_label, stats in frame_results.items():
#                 temp_dict[organs[int(elem)] + f"_{fr_label}_vol"] = volume * stats[0]
#                 temp_dict[organs[int(elem)] + f"_{fr_label}_min"] = stats[2]
#                 temp_dict[organs[int(elem)] + f"_{fr_label}_max"] = stats[1]
#                 temp_dict[organs[int(elem)] + f"_{fr_label}_mean"] = stats[3]
#                 temp_dict[organs[int(elem)] + f"_{fr_label}_std"] =  stats[4]
#         out_df.append(temp_dict)
#         # time Record
#         end_time = time.time()
#         end_datetime = datetime.datetime.now()
#         print(f"End Time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
#         elapsed_time_seconds = end_time - start_time
#         elapsed_timedelta = end_datetime - start_datetime
#         # time Record
#         print("Finished processing", idx)
#         print(f"Elapsed time: {elapsed_time_seconds:.2f} seconds")
#     long_bones_result_df = pd.DataFrame(out_df)
#     long_bones_result_df.to_csv(result_file_name, index=True)

# shpere bone

def sphere_bone_analyze(result_file_name = "sphere_bones_result_all.csv"):
    idx_list = os.listdir(".\\data\\")
    organs = om.get_organs()
    sphere_bones = [91]
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
                f"rt_skull_volume: {volume * rt_temp_out_report[0]:.2f}, "
                f"rt_skull_min: {rt_temp_out_report[2]:.2f}, "
                f"rt_skull_max: {rt_temp_out_report[1]:.2f}, "
                f"rt_skull_mean: {rt_temp_out_report[3]:.2f}, "
                f"rt_skull_std: {rt_temp_out_report[4]:.2f}"
                f"lt_skull_volume: {volume * lt_temp_out_report[0]:.2f}, "
                f"lt_skull_min: {lt_temp_out_report[2]:.2f}, "
                f"lt_skull_max: {lt_temp_out_report[1]:.2f}, "
                f"lt_skull_mean: {lt_temp_out_report[3]:.2f}, "
                f"lt_skull_std: {lt_temp_out_report[4]:.2f}"
            )
            temp_dict[organs[int(elem)]+"_rt_skull_range"] = ranges
            temp_dict[organs[int(elem)]+"_rt_skull_center_slice"] = center
            temp_dict[organs[int(elem)]+"_rt_skull_volume"] = volume * rt_temp_out_report[0]
            temp_dict[organs[int(elem)]+"_rt_skull_min"] = rt_temp_out_report[2]
            temp_dict[organs[int(elem)]+"_rt_skull_max"] = rt_temp_out_report[1]
            temp_dict[organs[int(elem)]+"_rt_skull_mean"] = rt_temp_out_report[3]
            temp_dict[organs[int(elem)]+"_rt_skull_std"] = rt_temp_out_report[4]
            temp_dict[organs[int(elem)]+"_lt_skull_volume"] = volume * lt_temp_out_report[0]
            temp_dict[organs[int(elem)]+"_lt_skull_min"] = lt_temp_out_report[2]
            temp_dict[organs[int(elem)]+"_lt_skull_max"] = lt_temp_out_report[1]
            temp_dict[organs[int(elem)]+"_lt_skull_mean"] = lt_temp_out_report[3]
            temp_dict[organs[int(elem)]+"_lt_skull_std"] = lt_temp_out_report[4]
        out_df.append(temp_dict)
        # time Record
        end_time = time.time()
        end_datetime = datetime.datetime.now()
        print(f"End Time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        elapsed_time_seconds = end_time - start_time
        # elapsed_timedelta = end_datetime - start_datetime
        # time Record
        print("Finished processing", idx)
        print(f"Elapsed time: {elapsed_time_seconds:.2f} seconds")
    sphere_bones_result_df = pd.DataFrame(out_df)
    sphere_bones_result_df.to_csv(result_file_name, index=True)

# cube bones


def cube_bone_analyze(result_file_name = "cube_bones_result_all.csv"):
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
            # axial range analysis
            axial_ranges = om.find_sig_lb_index_range(temp_lb_image)
            axial_ranges = max(axial_ranges, key=lambda x: x[1]-x[0])
            axial_center = int((axial_ranges[0]+3*axial_ranges[1])/4)
            # sagital range analysis
            sagital_ranges = om.find_axial_sig_lb_index_range(temp_lb_image, axial_index=1)
            sagital_ranges = max(sagital_ranges, key=lambda x: x[1]-x[0])
            sagital_center = int((sagital_ranges[0]+2*sagital_ranges[1])/3)
            # coronal range analysis
            coronal_ranges = om.find_axial_sig_lb_index_range(temp_lb_image, axial_index=2)
            coronal_ranges = max(coronal_ranges, key=lambda x: x[1]-x[0])
            coronal_center = int((coronal_ranges[0]+coronal_ranges[1])/2)
            # all center
            # 1 slice
            daxial, dsagital, dcoronal = 1, 1, 1
            filtered_3d_1fr = suv_nm_image[axial_center,
                                        sagital_center-dsagital:sagital_center+dsagital+1,
                                        coronal_center-dcoronal:coronal_center+dcoronal+1]
            sig_lb_size_1fr = temp_lb_image[axial_center,
                                        sagital_center-dsagital:sagital_center+dsagital+1,
                                        coronal_center-dcoronal:coronal_center+dcoronal+1]
            temp_1fr_filtered_suv_nm_image = filtered_3d_1fr * sig_lb_size_1fr
            non_zero_temp_1fr_image = temp_1fr_filtered_suv_nm_image[temp_1fr_filtered_suv_nm_image!=0]
            sum_volume_1fr = volume * len(non_zero_temp_1fr_image)
            result_min_1fr = np.min(non_zero_temp_1fr_image)
            result_max_1fr = np.max(non_zero_temp_1fr_image)
            result_mean_1fr = np.mean(non_zero_temp_1fr_image)
            result_std_1fr = np.std(non_zero_temp_1fr_image)
            # 3 slice
            daxial, dsagital, dcoronal = 1, 1, 1
            filtered_3d_3fr = suv_nm_image[axial_center-daxial:axial_center+daxial+1,
                                        sagital_center-dsagital:sagital_center+dsagital+1,
                                        coronal_center-dcoronal:coronal_center+dcoronal+1]
            sig_lb_size_3fr = temp_lb_image[axial_center-daxial:axial_center+daxial+1,
                                        sagital_center-dsagital:sagital_center+dsagital+1,
                                        coronal_center-dcoronal:coronal_center+dcoronal+1]
            temp_3fr_filtered_suv_nm_image = filtered_3d_3fr * sig_lb_size_3fr
            non_zero_temp_3fr_image = temp_3fr_filtered_suv_nm_image[temp_3fr_filtered_suv_nm_image!=0]
            sum_volume_3fr = volume * len(non_zero_temp_3fr_image)
            result_min_3fr = np.min(non_zero_temp_3fr_image)
            result_max_3fr = np.max(non_zero_temp_3fr_image)
            result_mean_3fr = np.mean(non_zero_temp_3fr_image)
            result_std_3fr = np.std(non_zero_temp_3fr_image)
            #######작성중
            print(
                f"idx: {idx}, seg: {elem}, "
                f"1_fr_volume: {sum_volume_1fr:.2f}, "
                f"1_fr_min: {result_min_1fr:.2f}, "
                f"1_fr_max: {result_max_1fr:.2f}, "
                f"1_fr_mean: {result_mean_1fr:.2f}, "
                f"1_fr_std: {result_std_1fr:.2f}"
                f"3_fr_volume: {sum_volume_3fr:.2f}, "
                f"3_fr_min: {result_min_3fr:.2f}, "
                f"3_fr_max: {result_max_3fr:.2f}, "
                f"3_fr_mean: {result_mean_3fr:.2f}, "
                f"3_fr_std: {result_std_3fr:.2f}"
            )
            temp_dict[organs[int(elem)]+"_1fr_center_pos"] = (axial_center, sagital_center, coronal_center)
            temp_dict[organs[int(elem)]+"_1fr_vol"] = sum_volume_1fr
            temp_dict[organs[int(elem)]+"_1fr_min"] = result_min_1fr
            temp_dict[organs[int(elem)]+"_1fr_max"] = result_max_1fr
            temp_dict[organs[int(elem)]+"_1fr_mean"] = result_mean_1fr
            temp_dict[organs[int(elem)]+"_1fr_std"] = result_std_1fr
            # 3frame
            temp_dict[organs[int(elem)]+"_3fr_vol"] = sum_volume_3fr
            temp_dict[organs[int(elem)]+"_3fr_min"] = result_min_3fr
            temp_dict[organs[int(elem)]+"_3fr_max"] = result_max_3fr
            temp_dict[organs[int(elem)]+"_3fr_mean"] = result_mean_3fr
            temp_dict[organs[int(elem)]+"_3fr_std"] = result_std_3fr
        out_df.append(temp_dict)
        # time Record
        end_time = time.time()
        end_datetime = datetime.datetime.now()
        print(f"End Time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        elapsed_time_seconds = end_time - start_time
        elapsed_timedelta = end_datetime - start_datetime
        # time Record
        print("Finished processing", idx)
    cube_bones_result_df = pd.DataFrame(out_df)
    cube_bones_result_df.to_csv(result_file_name, index=True)

if __name__ == "__main__":
    long_bone_analyze()
    sphere_bone_analyze()
    cube_bone_analyze()
    print("All analyses completed successfully.")

# range = [(a,b)]
# 1st quarter of center q1c

temp_lb_image

q1c = int(a + (b-a)*0.25)
q2c = int(a + (b-a)*0.5)
q3c = int(a + (b-a)*0.75)

zq1c = center_of_mass(temp_lb_image[q1c])
zq2c = center_of_mass(temp_lb_image[q2c])
zq3c = center_of_mass(temp_lb_image[q3c])

zq1c = (q1c, int(zq1c[0]),int(zq1c[1]))
zq2c = (q2c, int(zq2c[0]),int(zq2c[1]))
zq3c = (q3c, int(zq3c[0]),int(zq3c[1]))


offset = 1
filtered_volume_q1c = suv_nm_image[zq1c[0]-offset:zq1c[0]+offset+1, zq1c[1]-offset:zq1c[1]+offset+1, zq1c[2]-offset:zq1c[2]+offset+1]
filtered_volume_q2c = suv_nm_image[zq2c[0]-offset:zq2c[0]+offset+1, zq2c[1]-offset:zq2c[1]+offset+1, zq2c[2]-offset:zq2c[2]+offset+1]
filtered_volume_q3c = suv_nm_image[zq3c[0]-offset:zq3c[0]+offset+1, zq3c[1]-offset:zq3c[1]+offset+1, zq3c[2]-offset:zq3c[2]+offset+1]
