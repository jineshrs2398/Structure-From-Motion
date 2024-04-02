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


    def get_feat_matches(self, img_pair, num_of_samples= -1):
        """
        Returns all feature matches b/w given image pair unless num_of_samples
        is provided in which case it randomly returns that many samples from the
        image features
        imputs:
            img_pair: (i,j)
            num_of_samples: int (If -1, return all)
        """

        ith_view, jth_view = img_pair
        i, j = ith_view -1, jth_view -1

        # Get features common in i and j
        idxs = np.where(
            np.logical_and(
                self.visiblility_matrix[:, i],
                self.visiblility_matrix[:, j]
            )
        )[0]

        # Get num_of_samples from common features
        if num_of_samples >0:
            idxs = np.random.sample(idxs, num_of_samples)  # num_of_smaples

        vi = [self.features_u[idxs, i], self.features_v[idxs, i]] #list(N, , N,)
        vj = [self.features_u[idxs, j], self.features_v[idxs, j]] #list(N, , N,)

        vi = np.vstack(vi).T
        vj = np.vstack(vj).T

        return vi, vj, idxs
    
    def remove_matches(self, img_pair, outlier_idxs) -> None:
        """
        inputs:
            img_pair: (i,j)
        """
        _, jth_view = img_pair
        j = jth_view -  1

        self.visiblility_matrix[outlier_idxs, j] = False

    def get_2d_to_3d_correspondences(self, ith_view):
        """
        inputs:
            ith_view: view for which we need 2D <-> 3D correspondences
        outputs:
            v: M x 2 - features
            X: M x 3 - corresponding world points
        """
        i = ith_view - 1 

        # get indices from world_points wher it is no nan
        indices_world = np.argwhere(~np.isnan(self.world_points[:,0])).flatten()

        #we will get indices for the ith_view from visibility_matrix
        indices_visibility = np.where(self.visiblility_matrix[:,i])[0]

        #find intersection of indices_world and indices_visibility
        indices = np.array(list(set(indices_world) & set(indices_visibility)))

        v = [self.features_u[indices, i], self.features_v[indices, i]]
        v = np.vstack(v).T

        X = self.world_points[indices]
        return v, X

    def add_world_points(self, world_points, indices):
        """
        inputs:
            world_points: M x 3
            indices: M,
        outputs:
            None
        """
        self.world_points[indices] = world_points