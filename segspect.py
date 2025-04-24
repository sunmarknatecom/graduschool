import os, pydicom, copy, dicom2nifti, shutil, cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage.color import label2rgb # pip install scikit-image

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
    # CT-NM 정렬 지점 찾기 (NM, CT 시작 인덱스 구하기)
    first_ct_location = next(iter(ct_slice_locations.values()))
    first_nm_location = next(iter(nm_slice_locations.values()))
    if nm_slice_locations[0] <= ct_slice_locations[0]:
        ct_start_index = 0
        nm_start_index = None
        min_diff = float('inf')
        for key, value in nm_slice_locations.items():
            diff = abs(first_ct_location - value)
            if diff < min_diff:
                min_diff = diff
                nm_start_index = key
    elif (nm_slice_locations[0] - ct_slice_locations[0]) <= 1.25:
        ct_start_index = 0
        nm_start_index = 0
    else:
        nm_start_index = 0
        ct_start_index = None
        min_diff = float('inf')
        for key, value in ct_slice_locations.items():
            diff = abs(first_nm_location - value)
            if diff < min_diff:
                min_diff = diff
                ct_start_index = key
    # ct_start_index와 nm_start_index
    # NM 앞부분 트리밍밍
    trim_head_nm_slices = {}
    found_nm_start = False
    if nm_start_index != 0:
        for key, value in nm_slice_locations.items():
            if found_nm_start:
                trim_head_nm_slices[key] = value
            if key == (nm_start_index - 1):
                found_nm_start = True
    else:
        trim_head_nm_slices = copy.copy(nm_slice_locations)
    trim_head_ct_slices = {}
    found_ct_start  = False
    if ct_start_index != 0:
        for key, value in ct_slice_locations.items():
            if found_ct_start:
                trim_head_ct_slices[key] = value
            if key == (ct_start_index - 1):
                found_ct_start = True
    else:
        trim_head_ct_slices = copy.copy(ct_slice_locations)
    # NM 슬라이스 정리
    num_ct_slices = len(trim_head_ct_slices)
    num_nm_slices = list(trim_head_nm_slices.keys())[-1] # ref1 마지막슬라이스 번호가 갯수에서 -1 
    ct_index = 0
    nm_index = 0
    nm_slices_to_remove = []
    # 수정시작
    list_ct_slice_locations = [[i, (k, v)] for i, (k, v) in enumerate(trim_head_ct_slices.items())]
    list_nm_slice_locations = [[i, (k, v)] for i, (k, v) in enumerate(trim_head_nm_slices.items())]
    while ct_index < num_ct_slices:
        try:
            diff = abs(list_ct_slice_locations[ct_index][1][1]-list_nm_slice_locations[nm_index][1][1])
            if diff >= 1.25:
                nm_slices_to_remove.append(list_nm_slice_locations[nm_index][1][0])
                nm_index +=1
                # print("     ", i, diff, list_nm_slice_locations[nm_index][1][0], nm_index, ct_index, iteration_count)
            else:
                ct_index +=1
                nm_index +=1
                # print("case2", i, diff, list_nm_slice_locations[nm_index][1][0], nm_index, ct_index, iteration_count)
        except:
            break
            # print("    3", "end CT", ct_index, "end NM", nm_index)
    ct_end_index = list_ct_slice_locations[ct_index-1][1][0]
    nm_end_index = list_nm_slice_locations[nm_index-1][1][0]
    # 매칭되는 CT 시작 슬라이스 번호 찾기
    #  num_nm_slices가 ref1에서 -1이 되어 다시 1을 더하여 복원
    final_skip_index = np.concatenate((np.arange(nm_start_index),np.array(nm_slices_to_remove), np.arange(nm_end_index+1,num_nm_slices+1)))
    final_skip_index = np.array(final_skip_index, dtype=np.int32)
    final_nm_slices = np.delete(nm_file_obj.pixel_array, final_skip_index, axis=0)
    if ct_start_index != 0:
        head_trim_ct_index = np.arange(0, ct_start_index, dtype=np.int32)
    else:
        head_trim_ct_index = np.array([], dtype=np.int32)
    if nm_end_index != num_nm_slices - 1:
        tail_trim_ct_index = np.arange(ct_end_index + 1, num_ct_slices, dtype=np.int32)
    else:
        tail_trim_ct_index = np.array([], dtype=np.int32)
    ct_skip_index = np.concatenate((head_trim_ct_index, tail_trim_ct_index))
    return {
        "First ID of NM": nm_start_index,
        "Last ID of NM": nm_end_index,
        "First ID of CT": ct_start_index,
        "Last ID of CT": ct_end_index,
        "Length of CT ID": ct_end_index - ct_start_index + 1,
        "Length of NM ID": np.shape(final_nm_slices)[0],
        "calculated lengh of NM ID": nm_end_index - nm_start_index + 1 - len(final_skip_index),
        "delete CT index":ct_skip_index,
        "IDs to delete": nm_slices_to_remove,
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

def realign_ct_image(ct_image, ct_skip_list):
    if len(ct_skip_list) != 0:
        return np.delete(ct_image, ct_skip_list, axis=0)
    else:
        return ct_image

def realign_lb_image(src_nm_image, lb_image, nm_start_index, nm_end_index, insert_locations):
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
def to_1RGB_image(src_images, color="R"):
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
        print("totaly error")
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

def multi_view(src_images, bone_id):
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

# def get_images(idx):
#     '''
#     return : raw_ct_image(np), raw_lb_image(np), ct_image(np), nm_image(np), lb_image(np)
#     '''
#     temp_ct_path, temp_nm_path, temp_lb_path = get_paths(idx)
#     temp_ct_objs = open_CT(temp_ct_path)
#     temp_nm_obj = open_NM(temp_nm_path)
#     temp_lb_image = open_LB(temp_lb_path)
#     raw_temp_ct_image, tr_temp_ct_image = transform_ct_image(temp_ct_objs, temp_nm_obj)
#     # to raw data
#     transform_vars = get_transform_var(temp_ct_objs, temp_nm_obj)
#     temp_skip_list = transform_vars["final result"]
#     temp_ct_skip_list = transform_vars["delete CT index"]
#     tr_temp_lb_image = transform_label(temp_ct_objs, temp_nm_obj, temp_lb_image)
#     re_nm_image = realign_nm_image(temp_nm_obj, temp_skip_list)
#     re_raw_ct_image = realign_ct_image(raw_temp_ct_image, temp_ct_skip_list)
#     re_raw_lb_image = realign_ct_image(temp_lb_image, temp_ct_skip_list)
#     re_tr_ct_image = realign_ct_image(tr_temp_ct_image, temp_ct_skip_list)
#     re_tr_lb_image = realign_ct_image(tr_temp_lb_image, temp_ct_skip_list)
#     return re_raw_ct_image, re_raw_lb_image, re_tr_ct_image, re_nm_image, re_tr_lb_image

def get_images(idx):
    '''
    return : raw_ct_image(np), raw_lb_image(np), ct_image(np), nm_image(np), lb_image(np)
    '''
    temp_ct_path, temp_nm_path, temp_lb_path = get_paths(idx)
    temp_ct_objs = open_CT(temp_ct_path)
    temp_nm_obj = open_NM(temp_nm_path)
    temp_nm_image = temp_nm_obj.pixel_array
    temp_lb_image = open_LB(temp_lb_path)
    raw_temp_ct_image, tr_temp_ct_image = transform_ct_image(temp_ct_objs, temp_nm_obj)
    # to raw data
    transform_vars = get_transform_var(temp_ct_objs, temp_nm_obj)
    nm_start_index = transform_vars["First ID of NM"]
    lb_start_index = transform_vars["First ID of CT"]
    temp_skip_list = transform_vars["final result"]
    tr_temp_lb_image = transform_label(temp_ct_objs, temp_nm_obj, temp_lb_image)
    rn_tr_lb_image = realign_lb_image(temp_nm_image,tr_temp_lb_image, nm_start_index, lb_start_index, temp_skip_list)
    return raw_temp_ct_image, temp_lb_image, tr_temp_ct_image, temp_nm_image, rn_tr_lb_image

print("IDX", "raw_ct_image", "raw_lb_image", "ct_image", "nm_image", "lb_image")
for elem in idx_list:
    raw_ct_image, raw_lb_image, ct_image, nm_image, lb_image = get_images(elem)
    print(elem, np.shape(raw_ct_image), np.shape(raw_lb_image), np.shape(ct_image), np.shape(nm_image), np.shape(lb_image))
    # color_ct_image = to_color_image(ct_image)
    # red_lb_image = to_red_image(lb_image)
    # #
    # red_nm_image = to_red_image(nm_image)
    # out_fusion_image = fusion_images(color_ct_image, red_nm_image)
    # plt.imshow(out_fusion_image[570])
    # plt.show()
    #
    # plt.show(block=False)
    # plt.pause(2)
    # plt.close()

bones_index = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117]
organs = {1 : "spleen", 2 : "kidney_right", 3 : "kidney_left", 4 : "gallbladder", 5 : "liver", 6 : "stomach", 7 : "pancreas", 8 : "adrenal_gland_right", 9 : "adrenal_gland_left", 10 : "lung_upper_lobe_left", 11 : "lung_lower_lobe_left", 12 : "lung_upper_lobe_right", 13 : "lung_middle_lobe_right", 14 : "lung_lower_lobe_right", 15 : "esophagus", 16 : "trachea", 17 : "thyroid_gland", 18 : "small_bowel", 19 : "duodenum", 20 : "colon", 21 : "urinary_bladder", 22 : "prostate", 23 : "kidney_cyst_left", 24 : "kidney_cyst_right", 25 : "sacrum", 26 : "vertebrae_S1", 27 : "vertebrae_L5", 28 : "vertebrae_L4", 29 : "vertebrae_L3", 30 : "vertebrae_L2", 31 : "vertebrae_L1", 32 : "vertebrae_T12", 33 : "vertebrae_T11", 34 : "vertebrae_T10", 35 : "vertebrae_T9", 36 : "vertebrae_T8", 37 : "vertebrae_T7", 38 : "vertebrae_T6", 39 : "vertebrae_T5", 40 : "vertebrae_T4", 41 : "vertebrae_T3", 42 : "vertebrae_T2", 43 : "vertebrae_T1", 44 : "vertebrae_C7", 45 : "vertebrae_C6", 46 : "vertebrae_C5", 47 : "vertebrae_C4", 48 : "vertebrae_C3", 49 : "vertebrae_C2", 50 : "vertebrae_C1", 51 : "heart", 52 : "aorta", 53 : "pulmonary_vein", 54 : "brachiocephalic_trunk", 55 : "subclavian_artery_right", 56 : "subclavian_artery_left", 57 : "common_carotid_artery_right", 58 : "common_carotid_artery_left", 59 : "brachiocephalic_vein_left", 60 : "brachiocephalic_vein_right", 61 : "atrial_appendage_left", 62 : "superior_vena_cava", 63 : "inferior_vena_cava", 64 : "portal_vein_and_splenic_vein", 65 : "iliac_artery_left", 66 : "iliac_artery_right", 67 : "iliac_vena_left", 68 : "iliac_vena_right", 69 : "humerus_left", 70 : "humerus_right", 71 : "scapula_left", 72 : "scapula_right", 73 : "clavicula_left", 74 : "clavicula_right", 75 : "femur_left", 76 : "femur_right", 77 : "hip_left", 78 : "hip_right", 79 : "spinal_cord", 80 : "gluteus_maximus_left", 81 : "gluteus_maximus_right", 82 : "gluteus_medius_left", 83 : "gluteus_medius_right", 84 : "gluteus_minimus_left", 85 : "gluteus_minimus_right", 86 : "autochthon_left", 87 : "autochthon_right", 88 : "iliopsoas_left", 89 : "iliopsoas_right", 90 : "brain", 91 : "skull", 92 : "rib_left_1", 93 : "rib_left_2", 94 : "rib_left_3", 95 : "rib_left_4", 96 : "rib_left_5", 97 : "rib_left_6", 98 : "rib_left_7", 99 : "rib_left_8", 100 : "rib_left_9", 101 : "rib_left_10", 102 : "rib_left_11", 103 : "rib_left_12", 104 : "rib_right_1", 105 : "rib_right_2", 106 : "rib_right_3", 107 : "rib_right_4", 108 : "rib_right_5", 109 : "rib_right_6", 110 : "rib_right_7", 111 : "rib_right_8", 112 : "rib_right_9", 113 : "rib_right_10", 114 : "rib_right_11", 115 : "rib_right_12", 116 : "sternum", 117 : "costal_cartilages"}

idx = "003"
ct_path, nm_path, lb_path = get_paths(idx)
ct_objs = open_CT(ct_path)
nm_obj = open_NM(nm_path)
nm_image = nm_obj.pixel_array
raw_ct_image, raw_lb_image, ct_image, nm_image, lb_image = get_images(idx)
transform_vars = get_transform_var(ct_objs, nm_obj) 
nm_start_index = transform_vars["First ID of NM"]
nm_end_index = transform_vars["Last ID of NM"]
lb_start_index = transform_vars["First ID of CT"]
temp_skip_list = transform_vars["final result"]

single_lb_images = {}
for elem_bone in bones_index:
    raw_1ch_lb_image = only_seg_lb_1ch_image(raw_lb_image, elem_bone)
    # single_lb_images.append(raw_1ch_lb_image)
    raw_1ch_lb_image = transform_label(ct_objs, nm_obj, raw_1ch_lb_image)
    raw_1ch_lb_image = realign_lb_image(nm_image, raw_1ch_lb_image, nm_start_index, nm_end_index, temp_skip_list)
    single_lb_images[elem_bone] = raw_1ch_lb_image

for i, (bone_id, image) in enumerate(single_lb_images.items()):
   temp_color_image = to_1RGB_image(image, color='G')
   temp_color_nm_image = 255- to_color_image(nm_image)
   out_image = fusion_images(temp_color_image, temp_color_nm_image)
   multi_view(out_image, bone_id)
    
