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
    num_nm_slices = int(nm_file_obj.NumberOfFrames)
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
    final_skip_index = np.array(final_skip_index, dtype=np.int32)
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
    offset_x, offset_y = round((ct_x0-nm_x0)/nm_ps), round((ct_y0-nm_y0)/nm_ps)
    start_x_a = max(0, -offset_x)
    start_y_a = max(0, -offset_y)
    end_x_a = min(target_shape_x, nm_width - offset_x)
    end_y_a = min(target_shape_y, nm_height - offset_y)
    start_x_b = max(0, offset_x)
    start_y_b = max(0, offset_y)
    end_x_b = start_x_b + (end_x_a - start_x_a)
    end_y_b = start_y_b + (end_y_a - start_y_a)
    ret_image = np.zeros((ct_frames, nm_width, nm_height), dtype=ct_slices[0].pixel_array.dtype)
    out_raw_ct_image = []
    for i, temp_slice in enumerate(ct_slices):
        temp_image = temp_slice.pixel_array
        out_raw_ct_image.append(temp_image)
        temp_ret_image = cv2.resize(temp_image, (target_shape_x, target_shape_y))
        ret_image[i, start_y_b:end_y_b, start_x_b:end_x_b] = temp_ret_image[start_y_a:end_y_a, start_x_a:end_x_a]
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
    #nm_x0, nm_y0 = nm_file_obj.ImagePositionPatient[0], nm_file_obj.ImagePositionPatient[1]
    target_shape_x, target_shape_y = round(ct_width * ct_ps / nm_ps), round(ct_height * ct_ps / nm_ps)
    offset_x, offset_y = round((ct_x0-nm_x0)/nm_ps), round((ct_y0-nm_y0)/nm_ps)
    ret_image = np.zeros((ct_frames, nm_width, nm_height), dtype=ct_slices[0].pixel_array.dtype)
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
    ret_image = np.zeros((ct_frames, nm_width, nm_height), dtype=ct_slices[0].pixel_array.dtype)
    for i, temp_slice in enumerate(label_image):
        temp_ret_image = cv2.resize(temp_slice, (target_shape_x, target_shape_y))
        temp_ret_image = temp_ret_image.astype(ret_image.dtype)
        ret_image[i, start_y_b:end_y_b, start_x_b:end_x_b] = temp_ret_image[start_y_a:end_y_a, start_x_a:end_x_a]
    return ret_image


# image process
def to_red_image(src_images):
    temp_image = np.array([cv2.normalize(elem, None, 0, 255, cv2.NORM_MINMAX) for elem in src_images],dtype=np.uint8)
    temp_image = np.array([cv2.cvtColor(elem, cv2.COLOR_GRAY2RGB) for elem in temp_image],dtype=np.uint8)
    temp_image[:,:,:,2]=0
    temp_image[:,:,:,1]=0
    return temp_image

def to_color_image(src_images):
    #normalize
    norm_images = np.array([cv2.normalize(elem, None, 0, 255, cv2.NORM_MINMAX) for elem in src_images],dtype=np.uint8)
    return np.array([cv2.cvtColor(elem, cv2.COLOR_GRAY2RGB) for elem in norm_images], dtype=np.uint8)

def only_seg_lb_image(src_lb_image, seg_n = 70):
    return (src_lb_image == seg_n).astype(np.uint8)*seg_n

def only_seg_lb_1ch_image(src_lb_image, seg_n = 70):
    return (src_lb_image == seg_n).astype(np.uint8)

def find_min_max_index(src_lb_image, seg_n = 70):
    indices = np.argwhere(src_lb_image == seg_n)
    return np.min(indices[:,0]), np.max(indices[:,0])

def fusion_images(src1, src2):
    if np.shape(src1) == np.shape(src2):
        temp_out_image = np.zeros_like(src1)
        for i, (temp_1_image, temp_2_image) in enumerate(zip(src1, src2)):
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


def get_images(idx):
    '''
    return : raw_ct_image(np), raw_lb_image(np), ct_image(np), nm_image(np), lb_image(np)
    '''
    temp_ct_path, temp_nm_path, temp_lb_path = get_paths(idx)
    temp_ct_objs = open_CT(temp_ct_path)
    temp_nm_obj = open_NM(temp_nm_path)
    temp_lb_image = open_LB(temp_lb_path)
    raw_temp_ct_image, tr_temp_ct_image = transform_ct_image(temp_ct_objs, temp_nm_obj)
    tr_temp_lb_image = transform_label(temp_ct_objs, temp_nm_obj, temp_lb_image)
    temp_variables = get_transform_var(temp_ct_objs, temp_nm_obj)
    temp_skip_list = temp_variables["final result"]
    st_no = temp_variables["First ID of CT"]
    ed_no = temp_variables["Last ID of CT"]
    re_nm_image = realign_nm_image(temp_nm_obj, temp_skip_list)
    return raw_temp_ct_image[st_no:ed_no+1], temp_lb_image[st_no:ed_no+1], tr_temp_ct_image[st_no:ed_no+1], re_nm_image, tr_temp_lb_image[st_no:ed_no+1]

print("IDX", "raw_ct_image", "raw_lb_image", "ct_image", "nm_image", "lb_image")
for elem in idx_list:
    raw_ct_image, raw_lb_image, ct_image, nm_image, lb_image = get_images(elem)
    print(elem, np.shape(raw_ct_image), np.shape(raw_lb_image), np.shape(ct_image), np.shape(nm_image), np.shape(lb_image))
    #red_lb_image = to_red_image(lb_image)
    color_ct_image = to_color_image(ct_image)
    red_nm_image = to_red_image(nm_image)
    out_fusion_image = fusion_images(color_ct_image, red_nm_image)
    plt.imshow(out_fusion_image[570])
    plt.show(block=False)
    plt.pause(2)
    plt.close()

# 033 index는 CT 는 325~2082, NM은 -6부터 1886까지이면CT는 짤리지 않는데, NM을 기준으로 CT를 잘라도 일치하지 않음 확인
