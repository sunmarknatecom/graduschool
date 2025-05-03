import numpy as np
import cv2, copy


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