import numpy as np
import cv2

def cvt_mono_image(src_images, color="R"):
    """
    src_images = list of images to be converted to RGB
    color = color channel to be kept (R, G, or B)
    Function to convert grayscale images to RGB format and keep only the specified color channel.
    Returns:
        A numpy array of images in RGB format with the specified color channel kept.
    """
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
        print("totally error")
    return temp_image



def cvt_color_image(src_images):
    """
    src_images = list of images to be converted to RGB
    Function to convert grayscale images to RGB format.
    Returns:
        A numpy array of images in RGB format.
    """
    #normalize
    norm_images = np.array([cv2.normalize(elem, None, 0, 255, cv2.NORM_MINMAX) for elem in src_images],dtype=np.uint8)
    return np.array([cv2.cvtColor(elem, cv2.COLOR_GRAY2RGB) for elem in norm_images], dtype=np.uint8)