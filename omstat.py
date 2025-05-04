import numpy as np
import omfileutil

def get_nm_vol_info(idx):
    """
    idx = index of the file
    Function to calculate the volume of the source NM image.
    Returns:
        volume: Volume of the source NM image
    """
    _, nm_path, _ = omfileutil.get_file_paths(idx)
    src_nm_file_obj = omfileutil.open_NM_obj(nm_path)
    return float(src_nm_file_obj[0x0028,0x0030].value[0])*float(src_nm_file_obj[0x0028,0x0030].value[1])*float(src_nm_file_obj[0x0018,0x0050].value)

def get_nm_stat_info(src_data, src_mask):
    """
    src_data = source data (e.g., SUV image)
    src_mask = source mask (e.g., binary mask image)
    Function to calculate statistics (volume, max, min, mean, std) of the source data within the specified mask.
    Returns:
        volume: Volume of the source data within the mask
        max_value: Maximum value of the source data within the mask
        min_value: Minimum value of the source data within the mask
        mean_value: Mean value of the source data within the mask
        std_value: Standard deviation of the source data within the mask
    """
    if len(np.unique(src_mask)) == 2:
        out_data = src_data[src_mask==1]
    else:
        out_data = np.zeros_like(src_data)
    return np.sum(src_mask), np.max(out_data), np.min(out_data), np.mean(out_data), np.std(out_data)