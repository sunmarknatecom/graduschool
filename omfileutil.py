import os, pydicom
import numpy as np
import nibabel as nib

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
    lb_path = os.path.join(".\\labels\\",idx+"_nifti_label.nii")
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
    return temp_out_lb_image.astype(np.uint8)



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

