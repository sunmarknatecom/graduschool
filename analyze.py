import orbismarci as om
import pandas as pd
import os, copy
import numpy as np
import time
import datetime
from scipy.ndimage import center_of_mass

# long bones

global_input_list = os.listdir(".\\data\\")
global_input_list = global_input_list[:10]

# long_bone_analyze(input_list = global_input_list[10:], result_file_name = "auto_long_bones_result_02.csv")

def long_bone_analyze(input_list = None, result_file_name = "auto_long_bones_result_01.csv"):
    idx_list = input_list
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
            distal_1third_frame = int(ranges[0] + (ranges[1]-ranges[0])*0.25)
            middle_frame = int(ranges[0] + (ranges[1]-ranges[0])*0.5)
            proxim_3third_frame = int(ranges[0] + (ranges[1]-ranges[0])*0.75)
            zdc = center_of_mass(temp_lb_image[distal_1third_frame])
            zdc = (distal_1third_frame, int(zdc[0]), int(zdc[1]))
            zmc = center_of_mass(temp_lb_image[middle_frame])
            zmc = (middle_frame, int(zmc[0]), int(zmc[1]))
            zpc = center_of_mass(temp_lb_image[proxim_3third_frame])
            zpc = (proxim_3third_frame, int(zpc[0]), int(zpc[1]))
            offset = 1
            volume_zdc = suv_nm_image[zdc[0]-offset:zdc[0]+offset+1, zdc[1]-offset:zdc[1]+offset+1, zdc[2]-offset:zdc[2]+offset+1]
            volume_zmc = suv_nm_image[zmc[0]-offset:zmc[0]+offset+1, zmc[1]-offset:zmc[1]+offset+1, zmc[2]-offset:zmc[2]+offset+1]
            volume_zpc = suv_nm_image[zpc[0]-offset:zpc[0]+offset+1, zpc[1]-offset:zpc[1]+offset+1, zpc[2]-offset:zpc[2]+offset+1]
            temp_out_zdc = [np.shape(volume_zdc)[0]*np.shape(volume_zdc)[1]*np.shape(volume_zdc)[2], np.max(volume_zdc), np.min(volume_zdc), np.mean(volume_zdc), np.std(volume_zdc)]
            temp_out_zmc = [np.shape(volume_zmc)[0]*np.shape(volume_zmc)[1]*np.shape(volume_zmc)[2], np.max(volume_zmc), np.min(volume_zmc), np.mean(volume_zmc), np.std(volume_zmc)]
            temp_out_zpc = [np.shape(volume_zpc)[0]*np.shape(volume_zpc)[1]*np.shape(volume_zpc)[2], np.max(volume_zpc), np.min(volume_zpc), np.mean(volume_zpc), np.std(volume_zpc)]
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
            temp_dict[organs[int(elem)]+"_centers"] = (zdc, zmc, zpc)
            frame_results = {
                "distal_frame": temp_out_zdc,
                "middle_frame": temp_out_zmc,
                "proxim_frame": temp_out_zpc
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

def sphere_bone_analyze(input_list = global_input_list, result_file_name = "sphere_bones_result_all.csv"):
    idx_list = input_list
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
            # center(Frame) (-2 -1 0 1 2)
            temp_rearr_lb_image[:center-1, :, :] = 0
            temp_rearr_lb_image[center+2:, :, :] = 0
            ranges2 = om.find_axial_sig_lb_index_range(temp_rearr_lb_image, axial_index=1)
            ranges2 = max(ranges2, key=lambda x: x[1]-x[0])
            center2 = int((ranges2[0]+ranges2[1])/2)
            temp_rearr_lb_image[:,:center2-1,:] = 0
            temp_rearr_lb_image[:,center2+2:,:] = 0
            ranges3 = om.find_axial_sig_lb_index_range(temp_rearr_lb_image, axial_index=2)
            rt_range = ranges3[0]
            lt_range = ranges3[1]
            rt_temp_rearr_lb_image = copy.copy(temp_rearr_lb_image)
            lt_temp_rearr_lb_image = copy.copy(temp_rearr_lb_image)
            rt_temp_rearr_lb_image[:,:,:rt_range[0]] = 0
            rt_temp_rearr_lb_image[:,:,rt_range[1]:] = 0
            lt_temp_rearr_lb_image[:,:,:lt_range[0]] = 0
            lt_temp_rearr_lb_image[:,:,lt_range[1]:] = 0
            rt_skull_center = center_of_mass(rt_temp_rearr_lb_image)
            lt_skull_center = center_of_mass(lt_temp_rearr_lb_image)
            # results
            offset = 1
            rt_volume = temp_lb_image[rt_skull_center[0]-offset:rt_skull_center[0]+offset+1, rt_skull_center[1]-offset:rt_skull_center[1]+offset+1, rt_skull_center[2]-offset:rt_skull_center[2]+offset+1]
            lt_volume = temp_lb_image[lt_skull_center[0]-offset:lt_skull_center[0]+offset+1, lt_skull_center[1]-offset:lt_skull_center[1]+offset+1, lt_skull_center[2]-offset:lt_skull_center[2]+offset+1]
            rt_filtered_3d = suv_nm_image[rt_skull_center[0]-offset:rt_skull_center[0]+offset+1, rt_skull_center[1]-offset:rt_skull_center[1]+offset+1, rt_skull_center[2]-offset:rt_skull_center[2]+offset+1]
            lt_filtered_3d = suv_nm_image[lt_skull_center[0]-offset:lt_skull_center[0]+offset+1, lt_skull_center[1]-offset:lt_skull_center[1]+offset+1, lt_skull_center[2]-offset:lt_skull_center[2]+offset+1]
            out_rt_volume = np.sum(rt_volume)
            out_rt_min = np.min(rt_filtered_3d)
            out_rt_max = np.max(rt_filtered_3d)
            out_rt_mean = np.mean(rt_filtered_3d)
            out_rt_std = np.std(rt_filtered_3d)
            out_lt_volume = np.sum(lt_volume)
            out_lt_min = np.min(lt_filtered_3d)
            out_lt_max = np.max(lt_filtered_3d)
            out_lt_mean = np.mean(lt_filtered_3d)
            out_lt_std = np.std(lt_filtered_3d)
            print(
                f"idx: {idx}, seg: {elem}, "
                f"rt_skull_volume: {out_rt_volume}, "
                f"rt_skull_min: {out_rt_min}, "
                f"rt_skull_max: {out_rt_max}, "
                f"rt_skull_mean: {out_rt_mean}, "
                f"rt_skull_std: {out_rt_std}"
                f"lt_skull_volume: {out_lt_volume}, "
                f"lt_skull_min: {out_lt_min}, "
                f"lt_skull_max: {out_lt_max}, "
                f"lt_skull_mean: {out_lt_max}, "
                f"lt_skull_std: {out_lt_std}"
            )
            temp_dict[organs[int(elem)]+"_rt_skull_range"] = ranges
            temp_dict[organs[int(elem)]+"_rt_skull_center"] = rt_skull_center
            temp_dict[organs[int(elem)]+"_rt_skull_volume"] = rt_volume
            temp_dict[organs[int(elem)]+"_rt_skull_min"] = out_rt_min
            temp_dict[organs[int(elem)]+"_rt_skull_max"] = out_rt_max
            temp_dict[organs[int(elem)]+"_rt_skull_mean"] = out_rt_mean
            temp_dict[organs[int(elem)]+"_rt_skull_std"] = out_rt_std
            temp_dict[organs[int(elem)]+"_lt_skull_range"] = ranges2
            temp_dict[organs[int(elem)]+"_lt_skull_volume"] = lt_volume
            temp_dict[organs[int(elem)]+"_lt_skull_min"] = out_lt_min
            temp_dict[organs[int(elem)]+"_lt_skull_max"] = out_lt_max
            temp_dict[organs[int(elem)]+"_lt_skull_mean"] = out_lt_mean
            temp_dict[organs[int(elem)]+"_lt_skull_std"] = out_lt_std
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

global_input_list = os.listdir(".\\data\\")

for i in range(24):
    try:
        long_bone_analyze(input_list = global_input_list[i*10:(i+1)*10], result_file_name = f"auto_long_bones_result_{i+1:02d}.csv")
    except:
        print(f"Error processing batch {i+1}, skipping...")

for i in range(24):
    try:
        sphere_bone_analyze(input_list = global_input_list[i*10:(i+1)*10], result_file_name = f"auto_sphere_bones_result_{i+1:02d}.csv")
    except:
        print(f"Error processing batch {i+1}, skipping...")
