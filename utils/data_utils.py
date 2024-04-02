import csv
import glob
from typing import List
import os

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

class SFMMap():
    def __init__(self, path_to_matching_files: str) -> None:
        self.path = path_to_matching_files
        self.features_u, self.features_v = None, None
        self.visiblility_matrix = None
        self._load()

        self.world_points = np.empty((self.visiblility_matrix.shape[0], 3))
        self.world_points.fill(np.nan)

    def _load(self) -> None:
        """
        inputs:
            path:path to matching files
        output:
            features u coords: N x num_of_images
            features v coords: N x num_of_images
            visibility matrix: N x num_of_images
        """
        feats_u, feats_v, visibility_mat = [], [], []

        # get a list of all matching*.txt files
        matching_files = glob.glob(f"{self.path}/matching*.txt", recursive=False)
        match_file_names = []
        for match_file in matching_files:
            file_name = match_file.rsplit(".",1)[0][-1]
            match_file_names.append(file_name)

        num_images = len(match_file_names) + 1
        for ith_cam, match_file in zip(match_file_names, matching_files):
            with open(match_file) as file:
                reader = csv.reader(file, delimiter=' ')
                for row_idx, row in enumerate(reader):

                    feat_u = np.zeros((1, num_images))
                    feat_v = np.zeros((1, num_images))
                    visibility = np.zeros((1, num_images), dtype=bool)

                    # ignoring first the first line in each file
                    if row_idx == 0:
                        continue

                    n_matches_wrt_curr = int(row[0]) - 1

                    # convert camera number into array index
                    i = int(ith_cam) - 1

                    # read current feature coords
                    ui, vi = float(row[4]), float(row[5])

                    visibility[0,i] = True
                    feat_u[0, i] = ui
                    feat_v[0, i] = vi

                    #read j and subsequent feature coords in j
                    for idx in range(n_matches_wrt_curr):
                        jth_cam = row[3 * idx + 6]

                        # convert camera number to array index
                        j = int(jth_cam) - 1

                        uj = float(row[ 3 * idx + 7])
                        vj = float(row[ 3 * idx + 8])

                        visibility[0,i] = True
                        feat_u[0, j] = uj
                        feat_v[0, j] = vj

                    feats_u.append(feat_u)
                    feats_v.append(feat_v)
                    visibility_mat.append(visibility)

        self.features_u = np.vstack(feats_u)
        self.features_v = np.vstack(feats_v)
        self.visiblility_matrix = np.vstack(visibility_mat)


        