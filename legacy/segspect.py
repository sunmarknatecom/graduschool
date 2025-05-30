import os, pydicom, copy, cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap
from skimage.color import label2rgb # pip install scikit-image
import random

# global variables

idx_list = os.listdir(".\\data\\")

bones_index = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117]
organs = {1 : "spleen", 2 : "kidney_right", 3 : "kidney_left", 4 : "gallbladder", 5 : "liver", 6 : "stomach", 7 : "pancreas", 8 : "adrenal_gland_right", 9 : "adrenal_gland_left", 10 : "lung_upper_lobe_left", 11 : "lung_lower_lobe_left", 12 : "lung_upper_lobe_right", 13 : "lung_middle_lobe_right", 14 : "lung_lower_lobe_right", 15 : "esophagus", 16 : "trachea", 17 : "thyroid_gland", 18 : "small_bowel", 19 : "duodenum", 20 : "colon", 21 : "urinary_bladder", 22 : "prostate", 23 : "kidney_cyst_left", 24 : "kidney_cyst_right", 25 : "sacrum", 26 : "vertebrae_S1", 27 : "vertebrae_L5", 28 : "vertebrae_L4", 29 : "vertebrae_L3", 30 : "vertebrae_L2", 31 : "vertebrae_L1", 32 : "vertebrae_T12", 33 : "vertebrae_T11", 34 : "vertebrae_T10", 35 : "vertebrae_T9", 36 : "vertebrae_T8", 37 : "vertebrae_T7", 38 : "vertebrae_T6", 39 : "vertebrae_T5", 40 : "vertebrae_T4", 41 : "vertebrae_T3", 42 : "vertebrae_T2", 43 : "vertebrae_T1", 44 : "vertebrae_C7", 45 : "vertebrae_C6", 46 : "vertebrae_C5", 47 : "vertebrae_C4", 48 : "vertebrae_C3", 49 : "vertebrae_C2", 50 : "vertebrae_C1", 51 : "heart", 52 : "aorta", 53 : "pulmonary_vein", 54 : "brachiocephalic_trunk", 55 : "subclavian_artery_right", 56 : "subclavian_artery_left", 57 : "common_carotid_artery_right", 58 : "common_carotid_artery_left", 59 : "brachiocephalic_vein_left", 60 : "brachiocephalic_vein_right", 61 : "atrial_appendage_left", 62 : "superior_vena_cava", 63 : "inferior_vena_cava", 64 : "portal_vein_and_splenic_vein", 65 : "iliac_artery_left", 66 : "iliac_artery_right", 67 : "iliac_vena_left", 68 : "iliac_vena_right", 69 : "humerus_left", 70 : "humerus_right", 71 : "scapula_left", 72 : "scapula_right", 73 : "clavicula_left", 74 : "clavicula_right", 75 : "femur_left", 76 : "femur_right", 77 : "hip_left", 78 : "hip_right", 79 : "spinal_cord", 80 : "gluteus_maximus_left", 81 : "gluteus_maximus_right", 82 : "gluteus_medius_left", 83 : "gluteus_medius_right", 84 : "gluteus_minimus_left", 85 : "gluteus_minimus_right", 86 : "autochthon_left", 87 : "autochthon_right", 88 : "iliopsoas_left", 89 : "iliopsoas_right", 90 : "brain", 91 : "skull", 92 : "rib_left_1", 93 : "rib_left_2", 94 : "rib_left_3", 95 : "rib_left_4", 96 : "rib_left_5", 97 : "rib_left_6", 98 : "rib_left_7", 99 : "rib_left_8", 100 : "rib_left_9", 101 : "rib_left_10", 102 : "rib_left_11", 103 : "rib_left_12", 104 : "rib_right_1", 105 : "rib_right_2", 106 : "rib_right_3", 107 : "rib_right_4", 108 : "rib_right_5", 109 : "rib_right_6", 110 : "rib_right_7", 111 : "rib_right_8", 112 : "rib_right_9", 113 : "rib_right_10", 114 : "rib_right_11", 115 : "rib_right_12", 116 : "sternum", 117 : "costal_cartilages"}

# file_util

def get_file_paths(idx, root_path = ".\\data\\"):
    """
    idx = index of patient
    root_path = path to the data folder
    Function to get the file paths of CT, NM, and label files for a given patient index.
    Returns:
        ct_path: path to the CT file
        nm_path: path to the NM file
        lb_path: path to the label file
    * Note:
     - The function assumes that the CT and NM files are named with "_CT_" and "_NM_" respectively.
     - The function assumes that the label file is named with the pt index followed by "_nifti_label.nii".
     - The function assumes that the label files are located in a separate folder named "labels".
     - The function assumes that the CT and NM files are located in a folder named with the patient index.
     - The function assumes that the CT and NM files are in DICOM format.
     - The function assumes that the label files are in NIfTI format.
     - Note: The function assumes that the CT and NM files are in a folder named with the pt index.
     - Note: The function assumes that the label files are in a folder named "labels".
     - Note: The function assumes that the CT and NM files are in a folder named with the patient index.
     - ct : computed tomography
     - nm : nuclear medicine
     - lb : label
    """
    ct_path = os.path.join(root_path, idx, next((elem for elem in os.listdir(os.path.join(root_path,idx)) if "_CT_" in elem), None))
    nm_path = os.path.join(root_path, idx, next((elem for elem in os.listdir(os.path.join(root_path,idx)) if "_NM_" in elem), None))
    lb_path = ".\\labels\\"+idx+"_nifti_label.nii"
    #print(ct_path, nm_path)
    return ct_path, nm_path, lb_path



def open_CT_obj(folder_path = ".//TEST_CT//"):
    """
    folder_path = path to the folder containing CT DICOM files
    Function to open and read DICOM files in the specified folder.
    Returns:
        A list of DICOM objects sorted by slice location or image position.
    """
    temp_objs = [pydicom.dcmread(os.path.join(folder_path,elem)) for elem in os.listdir(folder_path)]
    if temp_objs[0].SliceLocation != False:
        temp_objs.sort(key=lambda x: float(x.SliceLocation))
    else:
        temp_objs.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return temp_objs



def open_NM_obj(folder_path = ".//TEST_NM//"):
    """
    folder_path = path to the folder containing NM DICOM files
    Function to open and read DICOM files in the specified folder.
    Returns:
        A DICOM object of the first file in the folder.
    """
    return pydicom.dcmread(os.path.join(folder_path, os.listdir(folder_path)[0]))



def load_CT_image(src_ct_file_objs):
    """
    ct_objs = list of CT DICOM objects
    Function to load and process CT images from the given DICOM objects.
    Returns:
        A numpy array of processed CT images.        
    """
    return np.array(np.array([elem.pixel_array for elem in src_ct_file_objs])*float(src_ct_file_objs[0].RescaleSlope)+float(src_ct_file_objs[0].RescaleIntercept),dtype=np.int16)



def load_LB_image(folder_path = ".//TEST_LB//"):
    """
    folder_path = path to the folder containing label files
    Function to load and process label images from the given folder.
    Returns:
        A numpy array of processed label images.
    """
    temp_lb_file_obj = nib.load(folder_path)
    temp_lb_image = temp_lb_file_obj.get_fdata()
    temp_out_lb_image = np.transpose(temp_lb_image, (2, 1, 0))
    temp_out_lb_image = np.flip(temp_out_lb_image, axis=1)
    return temp_out_lb_image



def load_NM_image(src_nm_file_obj):
    """
    src_nm_obj = NM DICOM object
    Function to load and process NM images from the given DICOM object.
    Returns:
        A numpy array of processed NM images.    
    """
    src_nm_images = src_nm_file_obj.pixel_array
    # Rescale intercept
    try:
        if (0x0028, 0x1052) in src_nm_file_obj:
            rescale_intercept = float(src_nm_file_obj[0x0028, 0x1052].value)
        else:
            rescale_intercept = float(src_nm_file_obj[0x0040, 0x9096][0][0x0040, 0x9224].value)
    except:
        print("No metadata for Rescale Intercept")
        rescale_intercept = 0.0
    # Rescale slope
    try:
        if (0x0028, 0x1053) in src_nm_file_obj:
            rescale_slope = float(src_nm_file_obj[0x0028, 0x1053].value)
        else:
            rescale_slope = float(src_nm_file_obj[0x0040, 0x9096][0][0x0040, 0x9225].value)
    except:
        print("No metadata for Rescale Slope")
        rescale_slope = 1.0
    scaled_image = src_nm_images * rescale_slope + rescale_intercept
    return scaled_image



def load_suv_nm_image(src_nm_file_obj):
    """
    src_nm_file_obj : NM DICOM object
    Function to convert NM images to SUV (Standardized Uptake Value) format.
    
    Returns:
        A numpy array of processed NM images in SUV format.    
    """
    scaled_image = load_NM_image(src_nm_file_obj)
    body_weight_kg = float(src_nm_file_obj[0x0010, 0x1030].value)
    injected_dose_bq = float(src_nm_file_obj[0x0054, 0x0016][0][0x0018, 0x1074].value)
    acquisition_time = float(src_nm_file_obj[0x0008, 0x0031].value)
    radiopharmaceutical_halflife_sec = float(src_nm_file_obj[0x0054, 0x0016][0][0x0018, 0x1075].value)
    injection_time = float(src_nm_file_obj[0x0054, 0x0016][0][0x0018, 0x1072].value)
    # Convert HHMMSS float time to seconds since midnight
    def time_to_seconds(time_val):
        hours = int(time_val // 10000)
        minutes = int((time_val % 10000) // 100)
        seconds = int(time_val % 100)
        return hours * 3600 + minutes * 60 + seconds
    acquisition_time_sec = time_to_seconds(acquisition_time)
    injection_time_sec = time_to_seconds(injection_time)
    time_diff_sec = acquisition_time_sec - injection_time_sec
    decay_correction_factor = (body_weight_kg * 1000 * 2 ** (time_diff_sec / radiopharmaceutical_halflife_sec)) / (injected_dose_bq * 1_000_000)
    dst_suv_image = scaled_image * decay_correction_factor
    return dst_suv_image


# image_processing

def get_align_info(src_ct_file_objs, src_nm_file_obj):
    """
    src_ct_file_objs = objects list of CT
    src_nm_file_obj = an object of NM
    Function to align CT and NM slices based on their slice locations.
    Returns:
        {
            "Start ID of NM": start_index_nm,
            "End ID of NM": end_index_nm,
            "Start ID of CT": start_index_ct,
            "End ID of CT": end_index_ct,
            "Length of CT ID": end_index_ct - start_index_ct + 1,
            "Length of NM ID": np.shape(filtered_nm_images)[0],
            "calculated lengh of NM ID": end_index_nm - start_index_nm + 1 - len(nm_indices_to_exclude),
            "delete CT index": ct_indices_to_exclude,
            "unmatched nm ID": unmatched_nm_indices,
            "final result": nm_indices_to_exclude
        }
    """
    # CT 데이터 처리
    img_shape_ct = list(src_ct_file_objs[0].pixel_array.shape)
    img_shape_ct.append(len(src_ct_file_objs))
    slice_locations_ct = {}
    for i, slice in enumerate(src_ct_file_objs):
        slice_locations_ct[i] = float(slice.SliceLocation)
    # NM 데이터 처리
    if "ImagePositionPatient" in src_nm_file_obj:
        start_position_nm = float(src_nm_file_obj["ImagePositionPatient"].value[2])  # 위치 정보
    else:
        start_position_nm = float(src_nm_file_obj["DetectorInformationSequence"][0]["ImagePositionPatient"].value[2])
    slice_locations_nm = {}
    slice_thickness_nm = float(src_nm_file_obj.SliceThickness)
    slices_num_nm = src_nm_file_obj.NumberOfFrames
    for i in range(slices_num_nm):
        slice_locations_nm[i] = float(start_position_nm + i * slice_thickness_nm)
    # CT-NM 정렬 지점 찾기 (NM, CT 시작 인덱스 구하기)
    first_location_ct = next(iter(slice_locations_ct.values()))
    first_location_nm = next(iter(slice_locations_nm.values()))
    if slice_locations_nm[0] <= slice_locations_ct[0]:
        start_index_ct = 0
        start_index_nm = None
        min_diff = float('inf')
        for key, value in slice_locations_nm.items():
            diff = abs(first_location_ct - value)
            if diff < min_diff:
                min_diff = diff
                start_index_nm = key
    elif (slice_locations_nm[0] - slice_locations_ct[0]) <= 1.25:
        start_index_ct = 0
        start_index_nm = 0
    else:
        start_index_nm = 0
        start_index_ct = None
        min_diff = float('inf')
        for key, value in slice_locations_ct.items():
            diff = abs(first_location_nm - value)
            if diff < min_diff:
                min_diff = diff
                start_index_ct = key
    # start_index_ct와 start_index_nm
    # NM 앞부분 트리밍
    trim_head_slices_nm = {}
    has_nm_start = False
    if start_index_nm != 0:
        for key, value in slice_locations_nm.items():
            if has_nm_start:
                trim_head_slices_nm[key] = value
            if key == (start_index_nm - 1):
                has_nm_start = True
    else:
        trim_head_slices_nm = copy.copy(slice_locations_nm)
    trim_head_objs_ct = {}
    has_ct_start  = False
    if start_index_ct != 0:
        for key, value in slice_locations_ct.items():
            if has_ct_start:
                trim_head_objs_ct[key] = value
            if key == (start_index_ct - 1):
                has_ct_start = True
    else:
        trim_head_objs_ct = copy.copy(slice_locations_ct)
    # NM 슬라이스 정리
    num_objs_ct = len(trim_head_objs_ct)
    slices_num_nm = list(trim_head_slices_nm.keys())[-1] # ref1 마지막슬라이스 번호가 갯수에서 -1 
    ct_index = 0
    nm_index = 0
    unmatched_nm_indices = []
    # 수정시작
    indexed_ct_slice_locations = [[i, (k, v)] for i, (k, v) in enumerate(trim_head_objs_ct.items())]
    indexed_nm_slice_locations = [[i, (k, v)] for i, (k, v) in enumerate(trim_head_slices_nm.items())]
    while ct_index < num_objs_ct:
        try:
            diff = abs(indexed_ct_slice_locations[ct_index][1][1]-indexed_nm_slice_locations[nm_index][1][1])
            if diff >= 1.25:
                unmatched_nm_indices.append(indexed_nm_slice_locations[nm_index][1][0])
                nm_index +=1
                # print("     ", i, diff, indexed_nm_slice_locations[nm_index][1][0], nm_index, ct_index, iteration_count)
            else:
                ct_index +=1
                nm_index +=1
                # print("case2", i, diff, indexed_nm_slice_locations[nm_index][1][0], nm_index, ct_index, iteration_count)
        except:
            break
            # print("    3", "end CT", ct_index, "end NM", nm_index)
    end_index_ct = indexed_ct_slice_locations[ct_index-1][1][0]
    end_index_nm = indexed_nm_slice_locations[nm_index-1][1][0]
    # 매칭되는 CT 시작 슬라이스 번호 찾기
    #  slices_num_nm가 ref1에서 -1이 되어 다시 1을 더하여 복원
    nm_indices_to_exclude = np.concatenate((np.arange(start_index_nm),np.array(unmatched_nm_indices), np.arange(end_index_nm+1,slices_num_nm+1)))
    nm_indices_to_exclude = np.array(nm_indices_to_exclude, dtype=np.int32)
    filtered_nm_images = np.delete(src_nm_file_obj.pixel_array, nm_indices_to_exclude, axis=0)
    if start_index_ct != 0:
        ct_head_indices_to_exclude = np.arange(0, start_index_ct, dtype=np.int32)
    else:
        ct_head_indices_to_exclude = np.array([], dtype=np.int32)
    if end_index_nm != slices_num_nm - 1:
        ct_tail_indices_to_exclude = np.arange(end_index_ct + 1, num_objs_ct, dtype=np.int32)
    else:
        ct_tail_indices_to_exclude = np.array([], dtype=np.int32)
    ct_indices_to_exclude = np.concatenate((ct_head_indices_to_exclude, ct_tail_indices_to_exclude))
    return {
        "Start ID of NM": start_index_nm,
        "End ID of NM": end_index_nm,
        "Start ID of CT": start_index_ct,
        "End ID of CT": end_index_ct,
        "Length of CT ID": end_index_ct - start_index_ct + 1,
        "Length of NM ID": np.shape(filtered_nm_images)[0],
        "calculated lengh of NM ID": end_index_nm - start_index_nm + 1 - len(nm_indices_to_exclude),
        "delete CT index": ct_indices_to_exclude,
        "unmatched nm ID": unmatched_nm_indices,
        "nm_indices_to_exclude": nm_indices_to_exclude
    }



def realign_ct_image(ct_image, ct_skip_list):
    """
    ct_image = CT image as a numpy array
    ct_skip_list = list of slices to remove from CT image
    Function to realign CT image by removing specified slices.
    Returns:
        Realigned CT image as a numpy array.
    """
    if len(ct_skip_list) != 0:
        return np.delete(ct_image, ct_skip_list, axis=0)
    else:
        return ct_image



def realign_nm_image(src_nm_file_obj, nm_slices_to_remove):
    """
    src_nm_file_obj = NM file object
    nm_slices_to_remove = list of slices to remove from NM image
    Function to realign NM image by removing specified slices.
    Returns:
        Realigned NM image as a numpy array.
    """
    temp_nm_image = src_nm_file_obj.pixel_array
    ret_image = np.delete(temp_nm_image, nm_slices_to_remove, axis=0)    
    return ret_image



def realign_lb_image(src_nm_image, lb_image, nm_start_index, nm_end_index, insert_locations):
    """
    src_nm_image = NM image as a numpy array
    lb_image = Label image as a numpy array
    nm_start_index = start index of NM image
    nm_end_index = end index of NM image
    insert_locations = list of locations to insert in the label image

    Function to realign label image by inserting specified locations.
    Returns:
        Realigned label image as a numpy array.
    """
    try:
        src_nm_index = list(range(nm_start_index, nm_end_index+1))
        lb_length_limit = len(lb_image)
        ret_insert_index = []
        for i, val in enumerate(src_nm_index):
            ret_insert_index.append(val)
            if val in insert_locations:
                ret_insert_index.append(val)
        ret_lb_image = np.zeros_like(src_nm_image, dtype=np.int16)
        ret_insert_index = np.array(ret_insert_index, dtype=np.int16)
        realign_lb_index = ret_insert_index - nm_start_index
        realign_lb_index = realign_lb_index[realign_lb_index < lb_length_limit]
        for ret_index, src_index in zip(ret_insert_index, realign_lb_index):
            ret_lb_image[ret_index]=lb_image[src_index]
        return ret_lb_image
    except:
        print("Not same frame image size between nm and lb")



def transform_ct_image(src_ct_file_obj, src_nm_file_obj):
    """
    src_ct_file_obj: list of CT DICOM objects
    src_nm_file_obj: NM file object

    Function to transform CT images to match the NM image size and position.

    Returns:
        A tuple containing:
        - A numpy array of raw CT images
        - A numpy array of transformed CT images resized to match the NM image size.
    """
    num_ct_slices = len(src_ct_file_obj)
    ct_img_height, ct_img_width = src_ct_file_obj[0].pixel_array.shape
    _, nm_img_height, nm_img_width = src_nm_file_obj.pixel_array.shape
    ct_pixel_spacing = float(src_ct_file_obj[0].PixelSpacing[0])
    nm_pixel_spacing = float(src_nm_file_obj.PixelSpacing[0])
    # CT Image Position
    if "ImagePositionPatient" in src_ct_file_obj[0]:
        ct_pos_x, ct_pos_y = float(src_ct_file_obj[0].ImagePositionPatient[0]), float(src_ct_file_obj[0].ImagePositionPatient[1])
    else:
        ct_pos_x, ct_pos_y = float(src_ct_file_obj[0]["DetectorInformationSequence"][0]["ImagePositionPatient"].value[0]), float(src_ct_file_obj[0]["DetectorInformationSequence"][0]["ImagePositionPatient"].value[1])
    # NM Image Position
    if "ImagePositionPatient" in src_nm_file_obj:
        nm_pos_x, nm_pos_y = float(src_nm_file_obj.ImagePositionPatient[0]), float(src_nm_file_obj.ImagePositionPatient[1])
    else:
        nm_pos_x, nm_pos_y = float(src_nm_file_obj["DetectorInformationSequence"][0]["ImagePositionPatient"].value[0]), float(src_nm_file_obj["DetectorInformationSequence"][0]["ImagePositionPatient"].value[1])
    resized_ct_width = round(ct_img_width * ct_pixel_spacing / nm_pixel_spacing)
    resized_ct_height = round(ct_img_height * ct_pixel_spacing / nm_pixel_spacing)
    offset_x = round((ct_pos_x - nm_pos_x) / nm_pixel_spacing)
    offset_y = round((ct_pos_y - nm_pos_y) / nm_pixel_spacing)
    ct_start_x = max(0, -offset_x)
    ct_start_y = max(0, -offset_y)
    ct_end_x = min(resized_ct_width, nm_img_width - offset_x)
    ct_end_y = min(resized_ct_height, nm_img_height - offset_y)
    nm_start_x = max(0, offset_x)
    nm_start_y = max(0, offset_y)
    nm_end_x = nm_start_x + (ct_end_x - ct_start_x)
    nm_end_y = nm_start_y + (ct_end_y - ct_start_y)
    aligned_ct_volume = np.zeros((num_ct_slices, nm_img_width, nm_img_height), dtype=src_ct_file_obj[0].pixel_array.dtype)
    raw_ct_volume = []
    for idx, ct_slice in enumerate(src_ct_file_obj):
        ct_image = ct_slice.pixel_array
        raw_ct_volume.append(ct_image)
        resized_ct_image = cv2.resize(ct_image, (resized_ct_width, resized_ct_height))
        aligned_ct_volume[idx, nm_start_y:nm_end_y, nm_start_x:nm_end_x] = resized_ct_image[ct_start_y:ct_end_y, ct_start_x:ct_end_x]
    return np.array(raw_ct_volume, dtype=np.int16), aligned_ct_volume



def transform_single_label(src_ct_file_objs, src_nm_file_obj, single_label_image):
    """
    src_ct_file_objs = list of CT DICOM objects
    src_nm_file_obj = NM file object
    label_image = label image as a numpy array
    Function to transform label images to match the NM image size and position.
    Returns:
        A numpy array of transformed label images resized to match the NM image size.
    """
    ct_width, ct_height = src_ct_file_objs[0].pixel_array.shape
    _, nm_width, nm_height = src_nm_file_obj.pixel_array.shape
    ct_ps = float(src_ct_file_objs[0].PixelSpacing[0])
    nm_ps = float(src_nm_file_obj.PixelSpacing[0])
    # CT 위치
    if "ImagePositionPatient" in src_ct_file_objs[0]:
        ct_x0, ct_y0 = float(src_ct_file_objs[0].ImagePositionPatient[0]), float(src_ct_file_objs[0].ImagePositionPatient[1])
    else:
        ct_x0, ct_y0 = float(src_ct_file_objs[0]["DetectorInformationSequence"][0]["ImagePositionPatient"].value[0]), float(src_ct_file_objs[0]["DetectorInformationSequence"][0]["ImagePositionPatient"].value[1])
    # NM 위치
    if "ImagePositionPatient" in src_nm_file_obj:
        nm_x0, nm_y0 = float(src_nm_file_obj.ImagePositionPatient[0]), float(src_nm_file_obj.ImagePositionPatient[1])
    else:
        nm_x0, nm_y0 = float(src_nm_file_obj["DetectorInformationSequence"][0]["ImagePositionPatient"].value[0]), float(src_nm_file_obj["DetectorInformationSequence"][0]["ImagePositionPatient"].value[1])
    #nm_x0, nm_y0 = src_nm_file_obj.ImagePositionPatient[0], src_nm_file_obj.ImagePositionPatient[1]
    target_shape_x, target_shape_y = round(ct_width * ct_ps / nm_ps), round(ct_height * ct_ps / nm_ps)
    offset_x, offset_y = round((ct_x0-nm_x0)/nm_ps), round((ct_y0-nm_y0)/nm_ps)
    # point들 정리
    # a를 변환전 배열, b를 변환 후 배열
    start_x_a = max(0, -offset_x)
    start_y_a = max(0, -offset_y)
    end_x_a = min(target_shape_x, nm_width - offset_x)
    end_y_a = min(target_shape_y, nm_height - offset_y)
    start_x_b = max(0, offset_x)
    start_y_b = max(0, offset_y)
    end_x_b = start_x_b + (end_x_a - start_x_a)
    end_y_b = start_y_b + (end_y_a - start_y_a)
    # 결과 이미지 초기화
    ret_image = np.zeros_like(src_nm_file_obj.pixel_array[0])
    temp_ret_image = cv2.resize(single_label_image, (target_shape_x, target_shape_y))
    temp_ret_image = temp_ret_image.astype(single_label_image.dtype)
    ret_image[start_y_b:end_y_b, start_x_b:end_x_b] = temp_ret_image[start_y_a:end_y_a, start_x_a:end_x_a]
    return ret_image

def transform_label(src_ct_file_objs, src_nm_file_obj, label_image):
    """
    src_ct_file_objs = list of CT DICOM objects
    src_nm_file_obj = NM file object
    label_image = label image as a numpy array
    Function to transform label images to match the NM image size and position.
    Returns:
        A numpy array of transformed label images resized to match the NM image size.
    """
    ct_frames = len(src_ct_file_objs)
    ct_width, ct_height = src_ct_file_objs[0].pixel_array.shape
    _, nm_width, nm_height = src_nm_file_obj.pixel_array.shape
    ct_ps = float(src_ct_file_objs[0].PixelSpacing[0])
    nm_ps = float(src_nm_file_obj.PixelSpacing[0])
    # CT 위치
    if "ImagePositionPatient" in src_ct_file_objs[0]:
        ct_x0, ct_y0 = float(src_ct_file_objs[0].ImagePositionPatient[0]), float(src_ct_file_objs[0].ImagePositionPatient[1])
    else:
        ct_x0, ct_y0 = float(src_ct_file_objs[0]["DetectorInformationSequence"][0]["ImagePositionPatient"].value[0]), float(src_ct_file_objs[0]["DetectorInformationSequence"][0]["ImagePositionPatient"].value[1])
    # NM 위치
    if "ImagePositionPatient" in src_nm_file_obj:
        nm_x0, nm_y0 = float(src_nm_file_obj.ImagePositionPatient[0]), float(src_nm_file_obj.ImagePositionPatient[1])
    else:
        nm_x0, nm_y0 = float(src_nm_file_obj["DetectorInformationSequence"][0]["ImagePositionPatient"].value[0]), float(src_nm_file_obj["DetectorInformationSequence"][0]["ImagePositionPatient"].value[1])
    #nm_x0, nm_y0 = src_nm_file_obj.ImagePositionPatient[0], src_nm_file_obj.ImagePositionPatient[1]
    target_shape_x, target_shape_y = round(ct_width * ct_ps / nm_ps), round(ct_height * ct_ps / nm_ps)
    offset_x, offset_y = round((ct_x0-nm_x0)/nm_ps), round((ct_y0-nm_y0)/nm_ps)
    # point들 정리
    # a를 변환전 배열, b를 변환 후 배열
    start_x_a = max(0, -offset_x)
    start_y_a = max(0, -offset_y)
    end_x_a = min(target_shape_x, nm_width - offset_x)
    end_y_a = min(target_shape_y, nm_height - offset_y)
    start_x_b = max(0, offset_x)
    start_y_b = max(0, offset_y)
    end_x_b = start_x_b + (end_x_a - start_x_a)
    end_y_b = start_y_b + (end_y_a - start_y_a)
    # 결과 이미지 초기화
    ret_image = np.zeros((ct_frames, nm_width, nm_height), dtype=src_ct_file_objs[0].pixel_array.dtype)
    for i, temp_slice in enumerate(label_image):
        temp_ret_image = cv2.resize(temp_slice, (target_shape_x, target_shape_y))
        temp_ret_image = temp_ret_image.astype(ret_image.dtype)
        ret_image[i, start_y_b:end_y_b, start_x_b:end_x_b] = temp_ret_image[start_y_a:end_y_a, start_x_a:end_x_a]
    return ret_image



# image process
def cvt_mono_image(src_images, color="R"):
    """
    src_images = list of images to be converted to RGB
    color = color channel to be kept (R, G, or B)
    Function to convert grayscale images to RGB format and keep only the specified color channel.
    Returns:
        A numpy array of images in RGB format with the specified color channel kept.
    """
    temp_image = np.array([cv2.normalize(elem, None, 0, 255, cv2.NORM_MINMAX) for elem in src_images],dtype=np.uint8)
    temp_image = np.array([cv2.cvtColor(elem, cv2.COLOR_GRAY2RGB) for elem in temp_image],dtype=np.uint8)
    try:
        if color == "R":
            temp_image[:,:,:,1]=0
            temp_image[:,:,:,2]=0
        elif color == "G":
            temp_image[:,:,:,0]=0
            temp_image[:,:,:,2]=0
        elif color == "B":
            temp_image[:,:,:,0]=0
            temp_image[:,:,:,1]=0
        else:
            print("only select in 'R' or 'G' or 'B'")
    except TypeError:
        print("totally error")
    return temp_image



def cvt_color_image(src_images):
    """
    src_images = list of images to be converted to RGB
    Function to convert grayscale images to RGB format.
    Returns:
        A numpy array of images in RGB format.
    """
    #normalize
    norm_images = np.array([cv2.normalize(elem, None, 0, 255, cv2.NORM_MINMAX) for elem in src_images],dtype=np.uint8)
    return np.array([cv2.cvtColor(elem, cv2.COLOR_GRAY2RGB) for elem in norm_images], dtype=np.uint8)



# label manipulation

def extract_raw_mask_label(src_lb_image, seg_n = 70):
    return (src_lb_image == seg_n).astype(np.uint8)*seg_n



def extract_binary_mask_label(src_lb_image, seg_n = 70):
    return (src_lb_image == seg_n).astype(np.uint8)



def find_min_max_index(src_lb_image, seg_n = 70):
    indices = np.argwhere(src_lb_image == seg_n)
    return np.min(indices[:,0]), np.max(indices[:,0])



def blend_images(src1_image, src2_image):
    if np.shape(src1_image) == np.shape(src2_image):
        temp_out_image = np.zeros_like(src1_image)
        for i, (temp_1_image, temp_2_image) in enumerate(zip(src1_image, src2_image)):
            temp_out_image[i] = cv2.addWeighted(temp_1_image, 0.5, temp_2_image, 0.5, 0)
        return temp_out_image
    else:
         print("error not equal shape")       



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



def merge_lb_image(src_ct_file_obj, src_nm_file_obj, raw_label_image, bone_indices):
    """
    Merges multiple binary label maps (one for each bone index) into a single labeled image.
    Args:
        src_ct_file_obj: List of CT DICOM objects.
        src_nm_file_obj: NM DICOM object.
        original_label_image: 3D label image with all bone labels.
        bone_indices: List of integer label values representing individual bones.
    Returns:
        A merged label image where each bone index occupies its respective region.
    """
    transformed_label_images = {}
    for bone_label in bone_indices:
        single_label_mask = extract_binary_mask_label(raw_label_image, bone_label)
        transformed_mask = transform_label(src_ct_file_obj, src_nm_file_obj, single_label_mask)
        transformed_label_images[bone_label] = transformed_mask * bone_label
    merged_label_image = np.zeros_like(transformed_mask)
    for bone_label, label_image in transformed_label_images.items():
        merged_label_image += label_image
    return merged_label_image


# visualization

def coloring_label(multi_label_image):
    return



def single_channel_view(src_images):
    frames, height, width = src_images.shape
    init_frame = 0
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    img_display = ax.imshow(src_images[init_frame])
    ax.set_title(f'Frame {init_frame}')
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, frames-1, valinit=init_frame, valstep=1)
    def update(val):
        frame = int(slider.val)
        img_display.set_data(src_images[frame])
        ax.set_title(f'Frame {frame}')
        fig.canvas.draw_idle()
    slider.on_changed(update)
    plt.show()



def multi_channel_view(src_images, bone_id):
    frames, height, width, channel = src_images.shape
    init_frame = 0
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    img_display = ax.imshow(src_images[init_frame])
    ax.set_title(f'{organs[bone_id]} CT Image')
    ax.set_title(f'{organs[bone_id]} Frame {init_frame}')
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, frames-1, valinit=init_frame, valstep=1)
    def update(val):
        frame = int(slider.val)
        img_display.set_data(src_images[frame])
        ax.set_title(f'{organs[bone_id]} Frame {frame}')
        fig.canvas.draw_idle()
    slider.on_changed(update)
    plt.show()



def merged_view(src_images):
    frames, height, width = src_images.shape
    unique_labels = np.unique(src_images)
    num_labels = len(unique_labels)
    colors = plt.get_cmap('tab20', num_labels)
    custom_colors = [(0, 0, 0, 1)] + [colors(i) for i in range(1, num_labels)]  # 0은 검은색
    custom_cmap = ListedColormap(custom_colors)
    init_frame = 0
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    img_display = ax.imshow(src_images[init_frame], cmap=custom_cmap)
    ax.set_title(f'NM Image')
    ax.set_title(f'Frame {init_frame}')
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, frames-1, valinit=init_frame, valstep=1)
    def update(val):
        frame = int(slider.val)
        img_display.set_data(src_images[frame])
        ax.set_title(f'Frame {frame}')
        fig.canvas.draw_idle()
    slider.on_changed(update)
    plt.show()



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
    bones_index = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117]
    organs = {1 : "spleen", 2 : "kidney_right", 3 : "kidney_left", 4 : "gallbladder", 5 : "liver", 6 : "stomach", 7 : "pancreas", 8 : "adrenal_gland_right", 9 : "adrenal_gland_left", 10 : "lung_upper_lobe_left", 11 : "lung_lower_lobe_left", 12 : "lung_upper_lobe_right", 13 : "lung_middle_lobe_right", 14 : "lung_lower_lobe_right", 15 : "esophagus", 16 : "trachea", 17 : "thyroid_gland", 18 : "small_bowel", 19 : "duodenum", 20 : "colon", 21 : "urinary_bladder", 22 : "prostate", 23 : "kidney_cyst_left", 24 : "kidney_cyst_right", 25 : "sacrum", 26 : "vertebrae_S1", 27 : "vertebrae_L5", 28 : "vertebrae_L4", 29 : "vertebrae_L3", 30 : "vertebrae_L2", 31 : "vertebrae_L1", 32 : "vertebrae_T12", 33 : "vertebrae_T11", 34 : "vertebrae_T10", 35 : "vertebrae_T9", 36 : "vertebrae_T8", 37 : "vertebrae_T7", 38 : "vertebrae_T6", 39 : "vertebrae_T5", 40 : "vertebrae_T4", 41 : "vertebrae_T3", 42 : "vertebrae_T2", 43 : "vertebrae_T1", 44 : "vertebrae_C7", 45 : "vertebrae_C6", 46 : "vertebrae_C5", 47 : "vertebrae_C4", 48 : "vertebrae_C3", 49 : "vertebrae_C2", 50 : "vertebrae_C1", 51 : "heart", 52 : "aorta", 53 : "pulmonary_vein", 54 : "brachiocephalic_trunk", 55 : "subclavian_artery_right", 56 : "subclavian_artery_left", 57 : "common_carotid_artery_right", 58 : "common_carotid_artery_left", 59 : "brachiocephalic_vein_left", 60 : "brachiocephalic_vein_right", 61 : "atrial_appendage_left", 62 : "superior_vena_cava", 63 : "inferior_vena_cava", 64 : "portal_vein_and_splenic_vein", 65 : "iliac_artery_left", 66 : "iliac_artery_right", 67 : "iliac_vena_left", 68 : "iliac_vena_right", 69 : "humerus_left", 70 : "humerus_right", 71 : "scapula_left", 72 : "scapula_right", 73 : "clavicula_left", 74 : "clavicula_right", 75 : "femur_left", 76 : "femur_right", 77 : "hip_left", 78 : "hip_right", 79 : "spinal_cord", 80 : "gluteus_maximus_left", 81 : "gluteus_maximus_right", 82 : "gluteus_medius_left", 83 : "gluteus_medius_right", 84 : "gluteus_minimus_left", 85 : "gluteus_minimus_right", 86 : "autochthon_left", 87 : "autochthon_right", 88 : "iliopsoas_left", 89 : "iliopsoas_right", 90 : "brain", 91 : "skull", 92 : "rib_left_1", 93 : "rib_left_2", 94 : "rib_left_3", 95 : "rib_left_4", 96 : "rib_left_5", 97 : "rib_left_6", 98 : "rib_left_7", 99 : "rib_left_8", 100 : "rib_left_9", 101 : "rib_left_10", 102 : "rib_left_11", 103 : "rib_left_12", 104 : "rib_right_1", 105 : "rib_right_2", 106 : "rib_right_3", 107 : "rib_right_4", 108 : "rib_right_5", 109 : "rib_right_6", 110 : "rib_right_7", 111 : "rib_right_8", 112 : "rib_right_9", 113 : "rib_right_10", 114 : "rib_right_11", 115 : "rib_right_12", 116 : "sternum", 117 : "costal_cartilages"}
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


if __name__ == "__main__":
    idx = "025"
    ct_path, nm_path, lb_path = get_file_paths(idx)
    ct_objs = open_CT_obj(ct_path)
    nm_obj = open_NM_obj(nm_path)
    nm_image = load_NM_image(nm_obj)
    raw_ct_image, raw_lb_image, ct_image, nm_image, suv_nm_image, lb_image = get_images(idx)
    transform_vars = get_align_info(ct_objs, nm_obj) 
    nm_start_index = transform_vars["Start ID of NM"]
    nm_end_index = transform_vars["End ID of NM"]
    lb_start_index = transform_vars["Start ID of CT"]
    temp_skip_list = transform_vars["nm_indices_to_exclude"]
    temp_lb_image = copy.copy(lb_image)
    temp_lb_image[temp_lb_image > 0] = 1
    out_image = nm_image * temp_lb_image
