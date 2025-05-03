import numpy as np
import omfileutil

def get_nm_vol_info(idx):
    _, nm_path, _ = omfileutil.get_file_paths(idx)
    src_nm_file_obj = omfileutil.open_NM_obj(nm_path)
    return float(src_nm_file_obj[0x0028,0x0030].value[0])*float(src_nm_file_obj[0x0028,0x0030].value[1])*float(src_nm_file_obj[0x0018,0x0050].value)

def get_nm_stat_info(src_data, src_mask):
    out_data = src_data[src_mask==1]
    return np.sum(src_mask), np.max(out_data), np.min(out_data), np.mean(out_data), np.std(out_data)