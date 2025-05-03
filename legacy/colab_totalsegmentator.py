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
