import cv2
import numpy as np
import glob
from Calibrator import Calibrator

calibrator = Calibrator(5, 7)
# calibrator.calibrate("./calibration_images/*", "./output_images/")
calibrator.load_parameters()

#image paths for stereo images



# Undistort benchmark image
# test_image = cv2.imread("./input_images/benchmark.jpg")
# distortion = cv2.undistort(test_image, calibrator.camera_matrix, calibrator.distortion_coeff, None, None)
# cv2.imwrite("./output_images/benchmark.jpg", test_image)



