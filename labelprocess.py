import numpy as np
import cv2
from arrayprocess import transform_label

bones_index = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117]
organs = {1 : "spleen", 2 : "kidney_right", 3 : "kidney_left", 4 : "gallbladder", 5 : "liver", 6 : "stomach", 7 : "pancreas", 8 : "adrenal_gland_right", 9 : "adrenal_gland_left", 10 : "lung_upper_lobe_left", 11 : "lung_lower_lobe_left", 12 : "lung_upper_lobe_right", 13 : "lung_middle_lobe_right", 14 : "lung_lower_lobe_right", 15 : "esophagus", 16 : "trachea", 17 : "thyroid_gland", 18 : "small_bowel", 19 : "duodenum", 20 : "colon", 21 : "urinary_bladder", 22 : "prostate", 23 : "kidney_cyst_left", 24 : "kidney_cyst_right", 25 : "sacrum", 26 : "vertebrae_S1", 27 : "vertebrae_L5", 28 : "vertebrae_L4", 29 : "vertebrae_L3", 30 : "vertebrae_L2", 31 : "vertebrae_L1", 32 : "vertebrae_T12", 33 : "vertebrae_T11", 34 : "vertebrae_T10", 35 : "vertebrae_T9", 36 : "vertebrae_T8", 37 : "vertebrae_T7", 38 : "vertebrae_T6", 39 : "vertebrae_T5", 40 : "vertebrae_T4", 41 : "vertebrae_T3", 42 : "vertebrae_T2", 43 : "vertebrae_T1", 44 : "vertebrae_C7", 45 : "vertebrae_C6", 46 : "vertebrae_C5", 47 : "vertebrae_C4", 48 : "vertebrae_C3", 49 : "vertebrae_C2", 50 : "vertebrae_C1", 51 : "heart", 52 : "aorta", 53 : "pulmonary_vein", 54 : "brachiocephalic_trunk", 55 : "subclavian_artery_right", 56 : "subclavian_artery_left", 57 : "common_carotid_artery_right", 58 : "common_carotid_artery_left", 59 : "brachiocephalic_vein_left", 60 : "brachiocephalic_vein_right", 61 : "atrial_appendage_left", 62 : "superior_vena_cava", 63 : "inferior_vena_cava", 64 : "portal_vein_and_splenic_vein", 65 : "iliac_artery_left", 66 : "iliac_artery_right", 67 : "iliac_vena_left", 68 : "iliac_vena_right", 69 : "humerus_left", 70 : "humerus_right", 71 : "scapula_left", 72 : "scapula_right", 73 : "clavicula_left", 74 : "clavicula_right", 75 : "femur_left", 76 : "femur_right", 77 : "hip_left", 78 : "hip_right", 79 : "spinal_cord", 80 : "gluteus_maximus_left", 81 : "gluteus_maximus_right", 82 : "gluteus_medius_left", 83 : "gluteus_medius_right", 84 : "gluteus_minimus_left", 85 : "gluteus_minimus_right", 86 : "autochthon_left", 87 : "autochthon_right", 88 : "iliopsoas_left", 89 : "iliopsoas_right", 90 : "brain", 91 : "skull", 92 : "rib_left_1", 93 : "rib_left_2", 94 : "rib_left_3", 95 : "rib_left_4", 96 : "rib_left_5", 97 : "rib_left_6", 98 : "rib_left_7", 99 : "rib_left_8", 100 : "rib_left_9", 101 : "rib_left_10", 102 : "rib_left_11", 103 : "rib_left_12", 104 : "rib_right_1", 105 : "rib_right_2", 106 : "rib_right_3", 107 : "rib_right_4", 108 : "rib_right_5", 109 : "rib_right_6", 110 : "rib_right_7", 111 : "rib_right_8", 112 : "rib_right_9", 113 : "rib_right_10", 114 : "rib_right_11", 115 : "rib_right_12", 116 : "sternum", 117 : "costal_cartilages"}

def label_info():
    """
    Function to print the label information of bones and organs.
    It iterates through the bones_index list and checks if each index is present in the organs dictionary.
    If found, it prints the index and the corresponding organ name. If not found, it prints "not found".
    This function is useful for understanding the mapping between bone indices and their corresponding organ names.
    Returns:
        None
    Example:
        label_info()
    # Output:
        1 : spleen
        2 : kidney_right
        3 : kidney_left
        4 : gallbladder
        5 : liver
        ...
    """
    for i in bones_index:
        if i in organs:
            print(f"{i} : {organs[i]}")
        else:
            print(f"{i} : not found")

def full_label_info():
    """
    Function to print the full label information of bones and organs.
    It iterates through the organs dictionary and prints each index and its corresponding organ name.
    This function is useful for understanding the complete mapping between bone indices and their corresponding organ names.
    Returns:
        None
    Example:
        full_label_info()
        # Output:
        # 1 : spleen
        # 2 : kidney_right
        # 3 : kidney_left
        ---
        """
    print("full label info")
    for k, v in organs.items():
        print(f"{k} : {v}")


def extract_raw_mask_label(src_lb_image, seg_n = 70):
    """
    src_lb_image = source label image
    seg_n = segment number to be extracted
    Function to extract a binary mask from the source label image based on the specified segment number.
    Returns:
        A binary mask where the specified segment number is set to 1 and all other values are set to 0.
    """
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

def find_sig_lb_index_range(binary_mask_lb_image):
    """
    Finds the start and end indices of non-zero elements in a binary mask label image.
    Args:
        binary_mask_lb_image: A binary mask label image (2D or 3D).
    Returns:
        A list of tuples, where each tuple contains the start and end indices of non-zero elements.
    Example:
        binary_mask_lb_image = np.array([[0, 0, 1], [1, 1, 0], [0, 0, 0]])
        ranges = find_sig_lb_index_range(binary_mask_lb_image)
        # ranges will be [(0, 2), (1, 3)]
    """
    ranges = []
    start = None
    arr = [(np.sum(elem) !=0).astype(int) for elem in binary_mask_lb_image]
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

def find_axial_sig_lb_index_range(binary_mask_lb_image, axial_index=1):
    """
    """
    ranges = []
    start = None
    axial_index_range = np.shape(binary_mask_lb_image)[axial_index]
    arr = [(np.sum(binary_mask_lb_image[:,idx,:]) !=0).astype(int) for idx in range(axial_index_range)]
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
    Example:
        merged_image = merge_lb_image(src_ct_file_obj, src_nm_file_obj, raw_label_image, bone_indices)
        # merged_image will contain the merged label image with each bone labeled accordingly.
    Comments: Lower numbers were overwritten by higher ones. Duplicates may need to be analyzed.
    """
    transformed_label_images = {}
    for bone_label in bone_indices:
        single_label_mask = extract_binary_mask_label(raw_label_image, bone_label)
        transformed_mask = transform_label(src_ct_file_obj, src_nm_file_obj, single_label_mask)
        transformed_label_images[bone_label] = transformed_mask * bone_label
    merged_label_image = np.zeros_like(transformed_mask)
    for bone_label, label_image in transformed_label_images.items():
        # print(bone_label, np.unique(label_image))
        if len(np.unique(label_image)) == 2:
            mask_value = int(np.unique(label_image)[1])
            merged_label_image[label_image == mask_value] = mask_value
        else:
            continue
    return merged_label_image
