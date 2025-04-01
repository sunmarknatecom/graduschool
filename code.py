import os, pydicom, copy
import numpy as np

idx = ".\\001"

idx_list = [".\\001", ".\\002", ".\\003" ".\\004"]

def get_paths(idx):
    ct_path = os.path.join(idx, next((elem for elem in os.listdir(idx) if "CT" in elem), None))
    nm_path = os.path.join(idx, next((elem for elem in os.listdir(idx) if "NM" in elem), None))
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
    nm_3d_image = nm_file_obj.pixel_array
    nm_3d_image_transposed = np.transpose(nm_3d_image, (1, 2, 0))
    nm_slice_locations = {}
    nm_slice_thickness = float(nm_file_obj.SliceThickness)
    num_nm_slices = np.shape(nm_3d_image_transposed)[2]
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
