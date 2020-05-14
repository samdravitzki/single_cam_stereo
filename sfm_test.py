import cv2
import numpy as np
import glob
from PIL import ExifTags
from PIL import Image
from matplotlib import pyplot
import numpy.linalg as la
from Calibrator import Calibrator, rectify_stereo_pair_uncalibrated

def downsample_image(image, reduce_factor):
    for i in range(0, reduce_factor):
        # If the image is grayscale get the size differently
        if len(image.shape) > 2:
            row, col = image.shape[:2]
        else:
            row, col = image.shape

        # Scale the image down by half each time
        image = cv2.pyrDown(image, dstsize=(col // 2, row // 2))

    return image

def main():
    calibrator = Calibrator(6, 9)
    calibrator.load_parameters()

    sfm_images = glob.glob("./sfm_images/*")

    for i in range(0, len(sfm_images) - 1):
        image = downsample_image(cv2.imread(sfm_images[i]), 3)
        next_image = downsample_image(cv2.imread(sfm_images[i+1]), 3)
        rectify_stereo_pair_uncalibrated(image, next_image, calibrator)


if __name__ == "__main__":
    main()