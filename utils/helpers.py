import numpy as np
def homogenize_coords(coords):
    """
    N x m -> N x (m+1)

    [[2, 3], [4, 5],[6, 7]] -> [[2, 3, 1],[4, 5, 1],[6, 7, 1]]
             
    """
    ret = np.concatenate((coords, np.ones(coords.shape[0],1)), axis=1)
    return ret