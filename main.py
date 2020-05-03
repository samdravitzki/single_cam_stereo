import cv2
import numpy as np
import glob
from Calibrator import Calibrator
from matplotlib import pyplot
from tkinter import *
from PIL import ImageTk, Image

win_size = 5
min_disp = -1
max_disp = 63  # min_disp * 9


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


class StereoDisparityMap:
    def __init__(self, left_image, right_image):
        self.left_image = left_image
        self.right_image = right_image

    def calculate_SRGB_disparity_map(self, win_size, min_disp, max_disp):

        num_disp = max_disp - min_disp

        if num_disp % 16 != 0:
            raise ArithmeticError("max_disp - min_disp is not divisble by 16")

        # Downsample the image to make them easier to work with
        image_left_downsampled = downsample_image(self.left_image, 3)
        image_right_downsampled = downsample_image(self.right_image, 3)

        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            disp12MaxDiff=1,
            P1=(8 * 3 * win_size ** 2),
            P2=(32 * 3 * win_size ** 2)
        )

        # compute the disparity map
        print("computing the disparity map...")
        disparity_map = stereo.compute(image_left_downsampled, image_right_downsampled)
        # pyplot.imshow(disparity_map, 'gray')
        # pyplot.show()
        return disparity_map

    def calculate_StereoBM_disparity_map(self, num_disparites, block_size):
        gray_left = cv2.cvtColor(self.left_image, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(self.right_image, cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM_create(num_disparites, block_size)
        disparity_map = stereo.compute(gray_left, gray_right)
        return disparity_map



calibrator = Calibrator(5, 7)
# calibrator.calibrate("./calibration_images/*", "./output_images/")
calibrator.load_parameters()

# image paths for stereo images
left_image_path = "./input_images/left.jpg"
right_image_path = "./input_images/right.jpg"

# load the images in
left_image = cv2.imread(left_image_path)
right_image = cv2.imread(right_image_path)
left_image_undistorted = calibrator.undistort_image(left_image)
right_image_undistorted = calibrator.undistort_image(right_image)

disparity_mapper = StereoDisparityMap(left_image_undistorted, right_image_undistorted)


def recalculate_SGBM():
    global photo
    global cv_img
    global size_entry
    global minimum_entry
    global maximum_entry


    cv_img = disparity_mapper.calculate_SRGB_disparity_map(
        int(size_entry.get()),
        int(minimum_entry.get()),
        int(maximum_entry.get()))
    photo = ImageTk.PhotoImage(image=Image.fromarray(cv_img))
    canvas.create_image(0, 0, image=photo, anchor=NW)

def recalculate_BM():
    global photo
    global cv_img

    cv_img = disparity_mapper.calculate_StereoBM_disparity_map(num_disparites=16, block_size=15)
    photo = ImageTk.PhotoImage(image=Image.fromarray(cv_img))
    canvas.create_image(0, 0, image=photo, anchor=NW)

window = Tk()
window.title("Disparity")

cv_img = cv2.cvtColor(cv2.imread("./output_images/disparity.jpg"), cv2.COLOR_BGR2RGB)

height, width = cv_img.shape[:2]

canvas = Canvas(window, width=width, height=height)
canvas.pack()

photo = ImageTk.PhotoImage(image=Image.fromarray(cv_img))

canvas.create_image(0, 0, image=photo, anchor=NW)

size = Label(window, text="Window Size")
size.pack(anchor=W)

size_entry = Entry(window)
size_entry.insert(0, "5")
size_entry.pack(anchor=W)

minimum = Label(window, text="Max Disparity")
minimum.pack(anchor=W)

minimum_entry = Entry(window)
minimum_entry.insert(0, "-1")
minimum_entry.pack(anchor=W)

maximum = Label(window, text="Min Disparity")
maximum.pack(anchor=W)

maximum_entry = Entry(window)
maximum_entry.insert(0, "15")
maximum_entry.pack(anchor=W)

btn_recalibrate = Button(window, text="Recalibrate SGBM", width=50, command=recalculate_SGBM)
btn_recalibrate.pack(anchor=CENTER, expand=True)

btn_recalibrate2 = Button(window, text="Recalibrate BM", width=50, command=recalculate_BM)
btn_recalibrate2.pack(anchor=CENTER, expand=True)

window.mainloop()




# root = Tk()
# my_gui = MyFirstGUI(root)
# root.mainloop()







# Undistort benchmark image
# test_image = cv2.imread("./input_images/benchmark.jpg")
# distortion = cv2.undistort(test_image, calibrator.camera_matrix, calibrator.distortion_coeff, None, None)
# cv2.imwrite("./output_images/benchmark.jpg", test_image)



