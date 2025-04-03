import os, pydicom, copy, dicom2nifti, shutil
import numpy as np

idx = ".\\001"

idx_list = [".\\001", ".\\002", ".\\003", ".\\004"]

# file_util

def get_paths(idx):
    ct_path = os.path.join(idx, next((elem for elem in os.listdir(idx) if "CT" in elem), None))
    nm_path = os.path.join(idx, next((elem for elem in os.listdir(idx) if "NM" in elem), None))
    print(ct_path, nm_path)
    return ct_path, nm_path

def open_CT(FolderPath = ".//TEST_CT//"):
    temp_objs = [pydicom.dcmread(os.path.join(FolderPath,elem)) for elem in os.listdir(FolderPath)]
    if temp_objs[0].SliceLocation != False:
        temp_objs.sort(key=lambda x: float(x.SliceLocation))
    else:
        temp_objs.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return temp_objs

def open_NM(FolderPath = ".//TEST_NM//"):
    return pydicom.dcmread(os.path.join(FolderPath, os.listdir(FolderPath)[0]))

# image process

def create_ct_image(ct_objs):
    return np.array([elem.pixel_array for elem in ct_objs])

def get_transform_var(ct_obj, nm_obj):
    """
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
    ct_slices = ct_obj
    img_shape_ct = list(ct_slices[0].pixel_array.shape)
    img_shape_ct.append(len(ct_slices))
    
    ct_slice_locations = {}
    for i, slice in enumerate(ct_slices):
        ct_slice_locations[i + 1] = float(slice.SliceLocation)
    # NM 데이터 처리
    nm_file_obj = nm_obj
    if "ImagePositionPatient" in nm_file_obj:
        nm_start_position = float(nm_file_obj["ImagePositionPatient"].value[2])  # 위치 정보
    else:
        nm_start_position = float(nm_file_obj["DetectorInformationSequence"][0]["ImagePositionPatient"].value[2])
    nm_slice_locations = {}
    nm_slice_thickness = float(nm_file_obj.SliceThickness)
    num_nm_slices = nm_file_obj.NumberOfFrames
    for i in range(num_nm_slices):
        nm_slice_locations[i + 1] = float(nm_start_position + i * nm_slice_thickness)
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
    if nm_start_index != 1:
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
    num_nm_slices = list(filtered_nm_slices.keys())[-1]
    ct_index = 1
    nm_index = list(filtered_nm_slices.keys())[0]
    nm_slices_to_remove = []
    iteration_count = 0
    while (ct_index <= num_ct_slices) and (nm_index <= num_nm_slices):
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
    return {
        "First ID of NM": nm_start_index,
        "Last ID of NM": nm_end_index,
        "First ID of CT": ct_start_index,
        "Last ID of CT": ct_end_index,
        "Length of CT ID": len(ct_slice_locations),
        "Length of NM ID": len(filtered_nm_slices),
        "IDs to delete": list(removed_nm_slices.keys())
    }

idx_list = [elem for elem in os.listdir() if os.path.isdir(elem)]

def main(idx_list):
    out_list = []
    for elem in idx_list:
        temp_ct_objs = open_CT(get_paths(elem)[0])
        temp_nm_obj = open_NM(get_paths(elem)[1])
        temp_vars = get_transform_var(temp_ct_objs, temp_nm_obj)
        valid_number = int(temp_vars["Last ID of NM"])-int(temp_vars["First ID of NM"])+1-len(temp_vars["IDs to delete"])
        print(elem, temp_vars)
        print(elem, temp_vars["Last ID of CT"], valid_number)
    

def convert_nifti(idx_list):
    for elem in idx_list:
        print(elem, " processing is stating")
        temp_ct_path = get_paths(elem)[0]
        os.mkdir(os.path.join(elem, elem + "_nifti"))
        temp_dst_path = os.path.normpath(os.path.join(elem, elem + "_nifti"))
        dicom2nifti.convert_directory(temp_ct_path, temp_dst_path)
        rn_src_name = os.path.join(temp_dst_path, os.listdir(temp_dst_path)[0])
        rn_dst_name = os.path.join(temp_dst_path, elem + "_nifiti.nii.gz")
        os.rename(rn_src_name, rn_dst_name)
        print(elem, " is complete")


for elem in idx_list:
    temp_src_path = os.path.join(elem, [elem for elem in os.listdir(elem) if "_nifti" in elem][0])
    file_name = os.listdir(temp_src_path)[0]
    temp_src_filename = os.path.join(temp_src_path, file_name)
    temp_dst_path = os.path.join("D:\\gradustudy\\uploadfiles\\",file_name)
    print(temp_src_filename, temp_dst_path)
    shutil.copyfile(temp_src_filename, temp_dst_path)


## colab

!pip install pydicom totalsegmentator

import os
from totalsegmentator.python_api import totalsegmentator

root_path = "/content/drive/MyDrive/gradstudy/nifti_ct"
dst_path = "/content/drive/MyDrive/gradstudy/result"

file_list = [os.path.join(root_path,elem) for elem in os.listdir(root_path)]

for elem in file_list:
    temp_dst_path =os.path.join(dst_path,os.path.basename(elem)[:4]+"nifti_label")
    print(elem)
    print(temp_dst_path)

for elem in file_list:
    temp_dst_path =os.path.join(dst_path,os.path.basename(elem)[:4]+"nifti_label")
    print(elem, "Processing start")
    print(temp_dst_path)
    totalsegmentator(elem, temp_dst_path, ml=True, task="total")
    print("Processing copmplete.")
