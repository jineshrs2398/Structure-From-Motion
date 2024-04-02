import argparse
from utils.data_utils import *
def main(args):
    base_path = args.basepath
    input_extension = ".png"
    calibration_file = args.calibrationFile
    
    #get images
    imgs =  load_images(base_path, input_extension)
    K = load_camera_intrinsics(f"{base_path}{calibration_file}")
    sfm_maps = SFMMap(base_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basepath', default='./Data/')
    parser.add_argument('--calibrationFile', default='calibration.txt')
    parser.add_argument('--display', action = 'store_true', help= "to display images")
    parser.add_argument('--debug', action = 'store_true', help= "to display debug information")

    args = parser.parse_args()
    main(args)