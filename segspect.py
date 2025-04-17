import os, pydicom, copy, dicom2nifti, shutil, cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

idx_list = os.listdir(".\\data\\")

# file_util

def get_paths(idx, root_path = ".\\data\\"):
    ct_path = os.path.join(root_path, idx, next((elem for elem in os.listdir(os.path.join(root_path,idx)) if "_CT_" in elem), None))
    nm_path = os.path.join(root_path, idx, next((elem for elem in os.listdir(os.path.join(root_path,idx)) if "_NM_" in elem), None))
    lb_path = ".\\labels\\"+idx+"_nifti_label.nii"
    #print(ct_path, nm_path)
    return ct_path, nm_path, lb_path

def open_CT(FolderPath = ".//TEST_CT//"):
    temp_objs = [pydicom.dcmread(os.path.join(FolderPath,elem)) for elem in os.listdir(FolderPath)]
    if temp_objs[0].SliceLocation != False:
        temp_objs.sort(key=lambda x: float(x.SliceLocation))
    else:
        temp_objs.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return temp_objs

def open_NM(FolderPath = ".//TEST_NM//"):
    return pydicom.dcmread(os.path.join(FolderPath, os.listdir(FolderPath)[0]))

def open_LB(FolderPath = ".//TEST_LB//"):
    temp_obj = nib.load(FolderPath)
    temp_image = temp_obj.get_fdata()
    temp_out_image = np.transpose(temp_image, (2, 1, 0))
    temp_out_image = np.flip(temp_out_image, axis=1)
    return temp_out_image
# image process

def create_ct_image(ct_objs):
    return np.array(np.array([elem.pixel_array for elem in ct_objs])*float(ct_objs[0].RescaleSlope)+float(ct_objs[0].RescaleIntercept),dtype=np.int16)

def get_transform_var(ct_slices, nm_file_obj):
    """
    ct_slices = objects list of CT
    nm_file_obj = an object of NM
    Function to align CT and NM slices based on their slice locations.
    Returns:
        {
            "First ID of NM": nm_start_index,
            "Last ID of NM": nm_end_index,
            "First ID of CT": ct_start_index,
            "Last ID of CT": ct_end_index,
            "Length of CT ID": len(ct_slice_locations),
            "Length of NM ID": len(filtered_nm_slices),
            "IDs to delete": list(removed_nm_slices.keys())
        }
    """
    # CT 데이터 처리
    img_shape_ct = list(ct_slices[0].pixel_array.shape)
    img_shape_ct.append(len(ct_slices))
    ct_slice_locations = {}
    for i, slice in enumerate(ct_slices):
        ct_slice_locations[i] = float(slice.SliceLocation)
    # NM 데이터 처리
    if "ImagePositionPatient" in nm_file_obj:
        nm_start_position = float(nm_file_obj["ImagePositionPatient"].value[2])  # 위치 정보
    else:
        nm_start_position = float(nm_file_obj["DetectorInformationSequence"][0]["ImagePositionPatient"].value[2])
    nm_slice_locations = {}
    nm_slice_thickness = float(nm_file_obj.SliceThickness)
    num_nm_slices = nm_file_obj.NumberOfFrames
    for i in range(num_nm_slices):
        nm_slice_locations[i] = float(nm_start_position + i * nm_slice_thickness)
    # CT-NM 정렬 지점 찾기 (NM 시작 인덱스 구하기)
    first_ct_location = next(iter(ct_slice_locations.values()))
    min_diff = float('inf')
    nm_start_index = None
    for key, value in nm_slice_locations.items():
        diff = abs(first_ct_location - value)
        if diff < min_diff:
            min_diff = diff
            nm_start_index = key
    # NM 슬라이스 필터링
    filtered_nm_slices = {}
    found_start = False
    if nm_start_index != 0:
        for key, value in nm_slice_locations.items():
            if found_start:
                filtered_nm_slices[key] = value
            if key == (nm_start_index - 1):
                found_start = True
    else:
        filtered_nm_slices = copy.copy(nm_slice_locations)
    # NM 슬라이스 정리
    removed_nm_slices = {}
    num_ct_slices = len(ct_slice_locations)
    num_nm_slices = list(filtered_nm_slices.keys())[-1] # ref1 마지막슬라이스 번호가 갯수에서 -1 
    ct_index = 0
    nm_index = list(filtered_nm_slices.keys())[0]
    nm_slices_to_remove = []
    iteration_count = 0
    while (ct_index < num_ct_slices) and (nm_index <= num_nm_slices):
        diff = ct_slice_locations[ct_index] - filtered_nm_slices[nm_index]
        if diff >= 1.23:
            nm_slices_to_remove.append(nm_index)
            removed_nm_slices[nm_index] = filtered_nm_slices[nm_index]
            nm_index += 1
            iteration_count += 1
        else:
            ct_index += 1
            nm_index += 1
            iteration_count += 1
    for elem in nm_slices_to_remove:
        del filtered_nm_slices[elem]
    # 매칭되는 CT 시작 슬라이스 번호 찾기
    first_nm_location = filtered_nm_slices[list(filtered_nm_slices.keys())[0]]
    ct_start_index = min(ct_slice_locations, key=lambda k: abs(ct_slice_locations[k] - first_nm_location))
    # 매칭되는 CT 마지막 슬라이스 번호 찾기
    last_nm_location = filtered_nm_slices[list(filtered_nm_slices.keys())[-1]]
    ct_end_index = min(ct_slice_locations, key=lambda k: abs(ct_slice_locations[k] - last_nm_location))
    # CT와 매칭되는 마지막 NM 슬라이스 번호 찾기
    nm_end_index = min(filtered_nm_slices, key=lambda k: abs(filtered_nm_slices[k] - ct_slice_locations[ct_end_index]))
    # num_nm_slices가 ref1에서 -1이 되어 다시 1을 더하여 복원
    final_skip_index = np.concatenate((np.arange(nm_start_index),np.array(list(removed_nm_slices.keys())), np.arange(nm_end_index+1,num_nm_slices+1)))
    return {
        "First ID of NM": nm_start_index,
        "Last ID of NM": nm_end_index,
        "First ID of CT": ct_start_index,
        "Last ID of CT": ct_end_index,
        "Length of CT ID": len(ct_slice_locations),
        "Length of NM ID": len(filtered_nm_slices),
        "IDs to delete": list(removed_nm_slices.keys()),
        "final result": final_skip_index
    }

def realign_nm_image(nm_file_obj, nm_slices_to_remove):
    """
    nm_file_obj = NM file object
    nm_slices_to_remove = list of slices to remove from NM image
    Function to realign NM image by removing specified slices.
    Returns:
        Realigned NM image as a numpy array.
    """
    temp_nm_image = nm_file_obj.pixel_array
    ret_image = np.delete(temp_nm_image, nm_slices_to_remove, axis=0)    
    return ret_image

def transform_ct_image(ct_slices, nm_file_obj):
    ct_frames = len(ct_slices)
    ct_width, ct_height = ct_slices[0].pixel_array.shape
    _, nm_width, nm_height = nm_file_obj.pixel_array.shape
    ct_ps = float(ct_slices[0].PixelSpacing[0])
    nm_ps = float(nm_file_obj.PixelSpacing[0])
    # CT 위치
    if "ImagePositionPatient" in ct_slices[0]:
        ct_x0, ct_y0 = float(ct_slices[0].ImagePositionPatient[0]), float(ct_slices[0].ImagePositionPatient[1])
    else:
        ct_x0, ct_y0 = float(ct_slices[0]["DetectorInformationSequence"][0]["ImagePositionPatient"].value[0]), float(ct_slices[0]["DetectorInformationSequence"][0]["ImagePositionPatient"].value[1])
    # NM 위치
    if "ImagePositionPatient" in nm_file_obj:
        nm_x0, nm_y0 = float(nm_file_obj.ImagePositionPatient[0]), float(nm_file_obj.ImagePositionPatient[1])
    else:
        nm_x0, nm_y0 = float(nm_file_obj["DetectorInformationSequence"][0]["ImagePositionPatient"].value[0]), float(nm_file_obj["DetectorInformationSequence"][0]["ImagePositionPatient"].value[1])
    #nm_x0, nm_y0 = nm_file_obj.ImagePositionPatient[0], nm_file_obj.ImagePositionPatient[1]
    target_shape_x, target_shape_y = round(ct_width * ct_ps / nm_ps), round(ct_height * ct_ps / nm_ps)
    offset_x, offset_y = round(abs((ct_x0-nm_x0)/nm_ps))-1, round(abs((ct_y0-nm_y0)/nm_ps))+1
    t_x_start = max(0, -offset_x)
    t_x_end = min(target_shape_x, target_shape_x-offset_x)
    t_y_start = max(0, -offset_y)
    t_y_end = min(target_shape_y, target_shape_y-offset_y)
    target_x_start = max(0, offset_x)
    target_x_end = target_x_start + (t_x_end - t_x_start)
    target_y_start = max(0, offset_y)
    target_y_end = target_y_start + (t_y_end - t_y_start)
    ret_image = np.zeros((ct_frames, nm_width, nm_height), dtype=ct_slices[0].pixel_array.dtype)
    out_raw_ct_image = []
    for i, temp_slice in enumerate(ct_slices):
        temp_image = temp_slice.pixel_array
        out_raw_ct_image.append(temp_image)
        temp_ret_image = cv2.resize(temp_image, (target_shape_x, target_shape_y))
        ret_image[i, target_y_start:target_y_end, target_x_start:target_x_end] = temp_ret_image[t_y_start:t_y_end,t_x_start:t_x_end]
    return np.array(out_raw_ct_image, dtype=np.int16), ret_image

def transform_label(ct_slices, nm_file_obj, label_image):
    ct_frames = len(ct_slices)
    ct_width, ct_height = ct_slices[0].pixel_array.shape
    _, nm_width, nm_height = nm_file_obj.pixel_array.shape
    ct_ps = float(ct_slices[0].PixelSpacing[0])
    nm_ps = float(nm_file_obj.PixelSpacing[0])
    # CT 위치
    if "ImagePositionPatient" in ct_slices[0]:
        ct_x0, ct_y0 = float(ct_slices[0].ImagePositionPatient[0]), float(ct_slices[0].ImagePositionPatient[1])
    else:
        ct_x0, ct_y0 = float(ct_slices[0]["DetectorInformationSequence"][0]["ImagePositionPatient"].value[0]), float(ct_slices[0]["DetectorInformationSequence"][0]["ImagePositionPatient"].value[1])
    # NM 위치
    if "ImagePositionPatient" in nm_file_obj:
        nm_x0, nm_y0 = float(nm_file_obj.ImagePositionPatient[0]), float(nm_file_obj.ImagePositionPatient[1])
    else:
        nm_x0, nm_y0 = float(nm_file_obj["DetectorInformationSequence"][0]["ImagePositionPatient"].value[0]), float(nm_file_obj["DetectorInformationSequence"][0]["ImagePositionPatient"].value[1])
    nm_x0, nm_y0 = nm_file_obj.ImagePositionPatient[0], nm_file_obj.ImagePositionPatient[1]
    target_shape_x, target_shape_y = round(ct_width * ct_ps / nm_ps), round(ct_height * ct_ps / nm_ps)
    t_x, t_y = round(abs((ct_x0-nm_x0)/nm_ps))-1, round(abs((ct_y0-nm_y0)/nm_ps))+1
    ret_image = np.zeros((ct_frames, nm_width, nm_height), dtype=ct_slices[0].pixel_array.dtype)
    for i, temp_slice in enumerate(label_image):
        temp_ret_image = cv2.resize(temp_slice, (target_shape_x, target_shape_y))
        ret_image[i, t_x:t_x+target_shape_x, t_y:t_y+target_shape_y] = temp_ret_image
    return ret_image


idx_list = [elem for elem in os.listdir() if os.path.isdir(elem)]




# 파일 처리

def main(idx_list):
    out_list = []
    for elem in idx_list:
        temp_ct_objs = open_CT(get_paths(elem)[0])
        temp_nm_obj = open_NM(get_paths(elem)[1])
        temp_vars = get_transform_var(temp_ct_objs, temp_nm_obj)
        valid_number = int(temp_vars["Last ID of NM"])-int(temp_vars["First ID of NM"])+1-len(temp_vars["IDs to delete"])
        print(elem, temp_vars)
        print(elem, temp_vars["Last ID of CT"], valid_number)
    
