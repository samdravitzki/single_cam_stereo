import cv2
import numpy as np
import glob
from PIL import ExifTags
from PIL import Image


class Calibrator:

    def __init__(self, row_count, col_count):
        self.row_count = row_count
        self.col_count = col_count

        self.ret = None
        self.camera_matrix = None
        self.distortion_coeff = None
        self.vecs = None
        self.focal_length = None


    def calibrate(self, input_directory, output_directory):
        chessboard = (self.row_count, self.col_count)
        critera = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

        # Empty object point
        objp = np.zeros((self.row_count * self.col_count, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.col_count, 0:self.row_count].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []

        # Select the directory containing the calibration images
        calibration_images = glob.glob(input_directory)
        gray_scale = None

        # iterate over image calculating the intrinsic matrix
        for i, image_src in enumerate(calibration_images):
            # Create a grayscale version of the image
            image = cv2.imread(image_src)
            gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Find the chessboard images corners
            success, corners = cv2.findChessboardCorners(gray_scale, chessboard, None)

            # If the corners are found
            if success:
                objpoints.append(objp)
                refined_corners = cv2.cornerSubPix(gray_scale, corners, (5, 5), (-1, -1), critera)
                imgpoints.append(refined_corners)  # append the refined camera matrix
                # draw corners on the image
                image = cv2.drawChessboardCorners(image, chessboard, refined_corners, success)
                cv2.imwrite("{0}draw-{1}.jpg".format(output_directory, i), image)

        # calibrate image based on the images
        ret, camera_matrix, distortion_coeff, rvecs, tvecs \
            = cv2.calibrateCamera(objpoints, imgpoints, gray_scale.shape[::-1], None, None)

        self.ret = ret
        self.camera_matrix = camera_matrix
        self.distortion_coeff = distortion_coeff
        self.vecs = (rvecs, tvecs)

        # Get the focal-point form the mobile image meta data

        exif_image = Image.open(calibration_images[0])

        # Select all meta data from the first image
        exif_data = {
            ExifTags.TAGS[k]: v
            for k, v in exif_image._getexif().items()
            if k in ExifTags.TAGS
        }

        # Select the focal length
        focal_length_exif = exif_data['FocalLength']

        self.focal_length = focal_length_exif[0] / focal_length_exif[1]
        self.save_parameters()

    def save_parameters(self):
        rvecs, tvecs = self.vecs
        np.save("./camera_parameters/ret", self.ret)
        np.save("./camera_parameters/K", self.camera_matrix)
        np.save("./camera_parameters/dist", self.distortion_coeff)
        np.save("./camera_parameters/rvecs", rvecs)
        np.save("./camera_parameters/tvecs", tvecs)
        np.save("./camera_parameters/focal_length", self.focal_length)


calibrator = Calibrator(5, 7)
calibrator.calibrate("./calibration_images/*", "./output_images/")

# Undistort benchmark image
test_image = cv2.imread("./input_images/benchmark.jpg")
distortion = cv2.undistort(test_image, calibrator.camera_matrix, calibrator.distortion_coeff, None, None)
cv2.imwrite("./output_images/benchmark.jpg", test_image)
