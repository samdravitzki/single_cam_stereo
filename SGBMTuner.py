import cv2
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import json
import time

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




# Global variables preset
imageToDisp = './scenes/dm-tune.jpg'
photo_width = 640
photo_height = 240
image_width = 320
image_height = 240

image_size = (image_width, image_height)

imgLeft = downsample_image(cv2.imread("./input_images/aloeL.jpg"), 1)
imgRight = downsample_image(cv2.imread("./input_images/aloeR.jpg"), 1)
# imgLeft = downsample_image(cv2.imread("./input_images/left3.jpg"), 3)
# imgRight = downsample_image(cv2.imread("./input_images/right3.jpg"), 3)


width_left, height_left = imgLeft.shape[:2]
width_right, height_right = imgRight.shape[:2]

if 0 in [width_left, height_left, width_right, height_right]:
    print("Error: Can't remap image.")

cv2.imshow('Left CALIBRATED', imgLeft)
cv2.imshow('Right CALIBRATED', imgRight)
cv2.waitKey(0)


class SGBMTuner:
    def __init__(self, imgL, imgR):
        self.rectified_pair = (imgL, imgR)
        self.MDIS = 16
        self.BS = 16
        self.WS = 3
        self.D12 = 1
        self.UR = 10
        self.SWS = 100
        self.SR = 32

        # Set up and draw interface
        # Draw left image and depth map
        axcolor = 'lightgoldenrodyellow'
        fig = plt.subplots(1, 2)
        plt.subplots_adjust(left=0.15, bottom=0.5)
        plt.subplot(1, 2, 1)
        self.dmObject = plt.imshow(self.rectified_pair[0], 'gray')

        # Building Depth Map for the first time
        disparity = self.stereo_depth_map(self.rectified_pair)

        plt.subplot(1, 2, 2)
        self.dmObject = plt.imshow(disparity, aspect='equal', cmap='jet')

        # Draw interface for adjusting parameters
        print('Start interface creation (it takes up to 30 seconds)...')

        MDISaxe = plt.axes([0.15, 0.01, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
        BSaxe = plt.axes([0.15, 0.05, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
        WSaxe = plt.axes([0.15, 0.09, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
        D12axe = plt.axes([0.15, 0.13, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
        URaxe = plt.axes([0.15, 0.17, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
        SWSaxe = plt.axes([0.15, 0.21, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
        SRaxe = plt.axes([0.15, 0.25, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height

        self.sMDIS = Slider(MDISaxe, 'minDisparity', 5.0, 255.0, valinit=5)
        self.sBS = Slider(BSaxe, 'blockSize', 5.0, 255.0, valinit=5)
        self.sWS = Slider(WSaxe, 'window_size', 5.0, 63.0, valinit=29)
        self.sD12 = Slider(D12axe, 'disp12MaxDiff', -100.0, 100.0, valinit=-25)
        self.sUR = Slider(URaxe, 'uniquenessRatio', 16.0, 256.0, valinit=128)
        self.sSWS = Slider(SWSaxe, 'speckleWindowSize', 0.0, 1000.0, valinit=100)
        self.sSR = Slider(SRaxe, 'speckleRange', 1.0, 20.0, valinit=10)

        self.sMDIS.on_changed(self.update)
        self.sBS.on_changed(self.update)
        self.sWS.on_changed(self.update)
        self.sD12.on_changed(self.update)
        self.sUR.on_changed(self.update)
        self.sSWS.on_changed(self.update)
        self.sSR.on_changed(self.update)

        print('Show interface to user')
        plt.show()

    def stereo_depth_map(self, rectified_pair):
        # c, r = rectified_pair[0].shape[:2]
        # disparity = np.zeros((c, r), np.uint8)
        block_matcher = cv2.StereoSGBM_create(
            minDisparity=self.MDIS,
            numDisparities=112-self.MDIS,
            blockSize=self.BS,
            uniquenessRatio=self.UR,
            speckleWindowSize=self.SWS,
            speckleRange=self.SR,
            disp12MaxDiff=self.D12,
            P1=8 * 3 * self.WS ** 2,
            P2=32 * 3 * self.WS ** 2,
        )
        # sbm.SADWindowSize = SWS
        dmLeft = cv2.cvtColor(rectified_pair[0], cv2.COLOR_BGR2GRAY)
        dmRight = cv2.cvtColor(rectified_pair[1], cv2.COLOR_BGR2GRAY)
        # cv2.FindStereoCorrespondenceBM(dmLeft, dmRight, disparity, sbm)
        disparity = block_matcher.compute(dmLeft, dmRight)
        # disparity_visual = cv.CreateMat(c, r, cv.CV_8U)
        local_max = disparity.max()
        local_min = disparity.min()
        print("MAX " + str(local_max))
        print("MIN " + str(local_min))
        disparity_visual = (disparity - local_min) * (1.0 / (local_max - local_min))
        local_max = disparity_visual.max()
        local_min = disparity_visual.min()
        print("MAX " + str(local_max))
        print("MIN " + str(local_min))
        # cv.Normalize(disparity, disparity_visual, 0, 255, cv.CV_MINMAX)
        # disparity_visual = np.array(disparity_visual)
        return disparity_visual

    # Update depth map parameters and redraw
    def update(self, val):

        self.MDIS = int(self.sMDIS.val / 2) * 2 + 1  # convert to ODD
        self.BS = int(self.sBS.val / 2) * 2 + 1
        self.WS = int(self.sWS.val / 2) * 2 + 1
        self.D12 = int(self.sD12.val)
        self.UR = int(self.sUR.val / 16) * 16
        self.SWS = int(self.sSWS.val)
        self.SR = int(self.sSR.val)
        print('Rebuilding depth map')
        disparity = self.stereo_depth_map(self.rectified_pair)
        self.dmObject.set_data(disparity)
        print('Redraw depth map')
        plt.draw()


stereo = SGBMTuner(imgLeft, imgRight)
