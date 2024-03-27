import csv
import glob
from typing import List

import cv2
import numpy as np

def load_images(path, extn = ".png"):
    """
    input:
        path - location from where files have to be loaded
        extn - file extension 
    output:
        images - list of images - N
    """
    img_files = glob.glob(f"{path}/*{extn}", recursive = False)

    imgs = [cv2.imread(img_file) for img_file in img_files]
    return imgs

def load_camera_intrinsics(path:str) -> List[List]:
    """
    Load camera intrinsics from a space-delimited text file.
    
    Args:
        path (str): Path to the file containing the camera intrinsic matrix.
    
    Returns:
        numpy.ndarray: A 3x3 camera intrinsic matrix.
    """
    K = np.loadtxt(path, delimiter=' ')
    return K