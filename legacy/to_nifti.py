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
