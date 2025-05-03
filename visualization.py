import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap

organs = {1 : "spleen", 2 : "kidney_right", 3 : "kidney_left", 4 : "gallbladder", 5 : "liver", 6 : "stomach", 7 : "pancreas", 8 : "adrenal_gland_right", 9 : "adrenal_gland_left", 10 : "lung_upper_lobe_left", 11 : "lung_lower_lobe_left", 12 : "lung_upper_lobe_right", 13 : "lung_middle_lobe_right", 14 : "lung_lower_lobe_right", 15 : "esophagus", 16 : "trachea", 17 : "thyroid_gland", 18 : "small_bowel", 19 : "duodenum", 20 : "colon", 21 : "urinary_bladder", 22 : "prostate", 23 : "kidney_cyst_left", 24 : "kidney_cyst_right", 25 : "sacrum", 26 : "vertebrae_S1", 27 : "vertebrae_L5", 28 : "vertebrae_L4", 29 : "vertebrae_L3", 30 : "vertebrae_L2", 31 : "vertebrae_L1", 32 : "vertebrae_T12", 33 : "vertebrae_T11", 34 : "vertebrae_T10", 35 : "vertebrae_T9", 36 : "vertebrae_T8", 37 : "vertebrae_T7", 38 : "vertebrae_T6", 39 : "vertebrae_T5", 40 : "vertebrae_T4", 41 : "vertebrae_T3", 42 : "vertebrae_T2", 43 : "vertebrae_T1", 44 : "vertebrae_C7", 45 : "vertebrae_C6", 46 : "vertebrae_C5", 47 : "vertebrae_C4", 48 : "vertebrae_C3", 49 : "vertebrae_C2", 50 : "vertebrae_C1", 51 : "heart", 52 : "aorta", 53 : "pulmonary_vein", 54 : "brachiocephalic_trunk", 55 : "subclavian_artery_right", 56 : "subclavian_artery_left", 57 : "common_carotid_artery_right", 58 : "common_carotid_artery_left", 59 : "brachiocephalic_vein_left", 60 : "brachiocephalic_vein_right", 61 : "atrial_appendage_left", 62 : "superior_vena_cava", 63 : "inferior_vena_cava", 64 : "portal_vein_and_splenic_vein", 65 : "iliac_artery_left", 66 : "iliac_artery_right", 67 : "iliac_vena_left", 68 : "iliac_vena_right", 69 : "humerus_left", 70 : "humerus_right", 71 : "scapula_left", 72 : "scapula_right", 73 : "clavicula_left", 74 : "clavicula_right", 75 : "femur_left", 76 : "femur_right", 77 : "hip_left", 78 : "hip_right", 79 : "spinal_cord", 80 : "gluteus_maximus_left", 81 : "gluteus_maximus_right", 82 : "gluteus_medius_left", 83 : "gluteus_medius_right", 84 : "gluteus_minimus_left", 85 : "gluteus_minimus_right", 86 : "autochthon_left", 87 : "autochthon_right", 88 : "iliopsoas_left", 89 : "iliopsoas_right", 90 : "brain", 91 : "skull", 92 : "rib_left_1", 93 : "rib_left_2", 94 : "rib_left_3", 95 : "rib_left_4", 96 : "rib_left_5", 97 : "rib_left_6", 98 : "rib_left_7", 99 : "rib_left_8", 100 : "rib_left_9", 101 : "rib_left_10", 102 : "rib_left_11", 103 : "rib_left_12", 104 : "rib_right_1", 105 : "rib_right_2", 106 : "rib_right_3", 107 : "rib_right_4", 108 : "rib_right_5", 109 : "rib_right_6", 110 : "rib_right_7", 111 : "rib_right_8", 112 : "rib_right_9", 113 : "rib_right_10", 114 : "rib_right_11", 115 : "rib_right_12", 116 : "sternum", 117 : "costal_cartilages"}

def coloring_label(multi_label_image):
    return

def single_channel_view(src_images):
    frames, height, width = src_images.shape
    init_frame = 0
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    img_display = ax.imshow(src_images[init_frame])
    ax.set_title(f'Frame {init_frame}')
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, frames-1, valinit=init_frame, valstep=1)
    def update(val):
        frame = int(slider.val)
        img_display.set_data(src_images[frame])
        ax.set_title(f'Frame {frame}')
        fig.canvas.draw_idle()
    slider.on_changed(update)
    plt.show()

def multi_channel_view(src_images, bone_id):
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

def merged_view(src_images):
    frames, height, width = src_images.shape
    unique_labels = np.unique(src_images)
    num_labels = len(unique_labels)
    colors = plt.get_cmap('tab20', num_labels)
    custom_colors = [(0, 0, 0, 1)] + [colors(i) for i in range(1, num_labels)]  # 0은 검은색
    custom_cmap = ListedColormap(custom_colors)
    init_frame = 0
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    img_display = ax.imshow(src_images[init_frame], cmap=custom_cmap)
    ax.set_title(f'NM Image')
    ax.set_title(f'Frame {init_frame}')
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, frames-1, valinit=init_frame, valstep=1)
    def update(val):
        frame = int(slider.val)
        img_display.set_data(src_images[frame])
        ax.set_title(f'Frame {frame}')
        fig.canvas.draw_idle()
    slider.on_changed(update)
    plt.show()