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

# imgL = downsample_image(cv2.imread("./input_images/aloeL.jpg"), 1)
# imgR = downsample_image(cv2.imread("./input_images/aloeR.jpg"), 1)
imgLeft = downsample_image(cv2.imread("./input_images/left3.jpg"), 3)
imgRight = downsample_image(cv2.imread("./input_images/right3.jpg"), 3)


width_left, height_left = imgLeft.shape[:2]
width_right, height_right = imgRight.shape[:2]

if 0 in [width_left, height_left, width_right, height_right]:
    print("Error: Can't remap image.")

cv2.imshow('Left CALIBRATED', imgLeft)
cv2.imshow('Right CALIBRATED', imgRight)
cv2.waitKey(0)


class StereoSMTuner:
    def __init__(self, imgL, imgR):
        self.rectified_pair = (imgL, imgR)
        self.SWS = 5
        self.PFS = 5
        self.PFC = 29
        self.MDS = -25
        self.NOD = 128
        self.TTH = 100
        self.UR = 10
        self.SR = 15
        self.SPWS = 100

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

        SWSaxe = plt.axes([0.15, 0.01, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
        PFSaxe = plt.axes([0.15, 0.05, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
        PFCaxe = plt.axes([0.15, 0.09, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
        MDSaxe = plt.axes([0.15, 0.13, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
        NODaxe = plt.axes([0.15, 0.17, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
        TTHaxe = plt.axes([0.15, 0.21, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
        URaxe = plt.axes([0.15, 0.25, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
        SRaxe = plt.axes([0.15, 0.29, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
        SPWSaxe = plt.axes([0.15, 0.33, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height

        self.sSWS = Slider(SWSaxe, 'SWS', 5.0, 255.0, valinit=5)
        self.sPFS = Slider(PFSaxe, 'PFS', 5.0, 255.0, valinit=5)
        self.sPFC = Slider(PFCaxe, 'PreFiltCap', 5.0, 63.0, valinit=29)
        self.sMDS = Slider(MDSaxe, 'MinDISP', -100.0, 100.0, valinit=-25)
        self.sNOD = Slider(NODaxe, 'NumOfDisp', 16.0, 256.0, valinit=128)
        self.sTTH = Slider(TTHaxe, 'TxtrThrshld', 0.0, 1000.0, valinit=100)
        self.sUR = Slider(URaxe, 'UnicRatio', 1.0, 20.0, valinit=10)
        self.sSR = Slider(SRaxe, 'SpcklRng', 0.0, 40.0, valinit=15)
        self.sSPWS = Slider(SPWSaxe, 'SpklWinSze', 0.0, 300.0, valinit=100)

        self.sSWS.on_changed(self.update)
        self.sPFS.on_changed(self.update)
        self.sPFC.on_changed(self.update)
        self.sMDS.on_changed(self.update)
        self.sNOD.on_changed(self.update)
        self.sTTH.on_changed(self.update)
        self.sUR.on_changed(self.update)
        self.sSR.on_changed(self.update)
        self.sSPWS.on_changed(self.update)

        print('Show interface to user')
        plt.show()

    def stereo_depth_map(self, rectified_pair):
        # c, r = rectified_pair[0].shape[:2]
        # disparity = np.zeros((c, r), np.uint8)
        sbm = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        # sbm.SADWindowSize = SWS
        sbm.setPreFilterType(1)
        sbm.setPreFilterSize(self.PFS)
        sbm.setPreFilterCap(self.PFC)
        sbm.setMinDisparity(self.MDS)
        sbm.setNumDisparities(self.NOD)
        sbm.setTextureThreshold(self.TTH)
        sbm.setUniquenessRatio(self.UR)
        sbm.setSpeckleRange(self.SR)
        sbm.setSpeckleWindowSize(self.SPWS)
        dmLeft = cv2.cvtColor(rectified_pair[0], cv2.COLOR_BGR2GRAY)
        dmRight = cv2.cvtColor(rectified_pair[1], cv2.COLOR_BGR2GRAY)
        # cv2.FindStereoCorrespondenceBM(dmLeft, dmRight, disparity, sbm)
        disparity = sbm.compute(dmLeft, dmRight)
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
        self.SWS = int(self.sSWS.val / 2) * 2 + 1  # convert to ODD
        self.PFS = int(self.sPFS.val / 2) * 2 + 1
        self.PFC = int(self.sPFC.val / 2) * 2 + 1
        self.MDS = int(self.sMDS.val)
        self.NOD = int(self.sNOD.val / 16) * 16
        self.TTH = int(self.sTTH.val)
        self.UR = int(self.sUR.val)
        self.SR = int(self.sSR.val)
        self.SPWS = int(self.sSPWS.val)
        print('Rebuilding depth map')
        disparity = self.stereo_depth_map(self.rectified_pair)
        self.dmObject.set_data(disparity)
        print('Redraw depth map')
        plt.draw()


stereo = StereoSMTuner(imgLeft, imgRight)

# rectified_pair = (imgL, imgR)

# # Depth map function
# SWS = 5
# PFS = 5
# PFC = 29
# MDS = -25
# NOD = 128
# TTH = 100
# UR = 10
# SR = 15
# SPWS = 100
#
#
# def stereo_depth_map(rectified_pair):
#     # c, r = rectified_pair[0].shape[:2]
#     # disparity = np.zeros((c, r), np.uint8)
#     sbm = cv2.StereoBM_create(numDisparities=16, blockSize=15)
#     # sbm.SADWindowSize = SWS
#     sbm.setPreFilterType(1)
#     sbm.setPreFilterSize(PFS)
#     sbm.setPreFilterCap(PFC)
#     sbm.setMinDisparity(MDS)
#     sbm.setNumDisparities(NOD)
#     sbm.setTextureThreshold(TTH)
#     sbm.setUniquenessRatio(UR)
#     sbm.setSpeckleRange(SR)
#     sbm.setSpeckleWindowSize(SPWS)
#     dmLeft = cv2.cvtColor(rectified_pair[0], cv2.COLOR_BGR2GRAY)
#     dmRight = cv2.cvtColor(rectified_pair[1], cv2.COLOR_BGR2GRAY)
#     # cv2.FindStereoCorrespondenceBM(dmLeft, dmRight, disparity, sbm)
#     disparity = sbm.compute(dmLeft, dmRight)
#     # disparity_visual = cv.CreateMat(c, r, cv.CV_8U)
#     local_max = disparity.max()
#     local_min = disparity.min()
#     print("MAX " + str(local_max))
#     print("MIN " + str(local_min))
#     disparity_visual = (disparity - local_min) * (1.0 / (local_max - local_min))
#     local_max = disparity_visual.max()
#     local_min = disparity_visual.min()
#     print("MAX " + str(local_max))
#     print("MIN " + str(local_min))
#     # cv.Normalize(disparity, disparity_visual, 0, 255, cv.CV_MINMAX)
#     # disparity_visual = np.array(disparity_visual)
#     return disparity_visual


# Set up and draw interface
# Draw left image and depth map
# axcolor = 'lightgoldenrodyellow'
# fig = plt.subplots(1, 2)
# plt.subplots_adjust(left=0.15, bottom=0.5)
# plt.subplot(1, 2, 1)
# dmObject = plt.imshow(rectified_pair[0], 'gray')

# saveax = plt.axes([0.3, 0.38, 0.15, 0.04])  # stepX stepY width height
# buttons = Button(saveax, 'Save settings', color=axcolor, hovercolor='0.975')
#
#
# def save_map_settings(event):
#     buttons.label.set_text("Saving...")
#     print('Saving to file...')
#     result = json.dumps({'SADWindowSize': SWS, 'preFilterSize': PFS, 'preFilterCap': PFC,
#                          'minDisparity': MDS, 'numberOfDisparities': NOD, 'textureThreshold': TTH,
#                          'uniquenessRatio': UR, 'speckleRange': SR, 'speckleWindowSize': SPWS},
#                         sort_keys=True, indent=4, separators=(',', ':'))
#     fName = '3dmap_set.txt'
#     f = open(str(fName), 'w')
#     f.write(result)
#     f.close()
#     buttons.label.set_text("Save to file")
#     print('Settings saved to file ' + fName)
#
#
# buttons.on_clicked(save_map_settings)
#
# loadax = plt.axes([0.5, 0.38, 0.15, 0.04])  # stepX stepY width height
# buttonl = Button(loadax, 'Load settings', color=axcolor, hovercolor='0.975')
#
#
# def load_map_settings(event):
#     global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS, loading_settings
#     loading_settings = 1
#     fName = '3dmap_set.txt'
#     print('Loading parameters from file...')
#     buttonl.label.set_text("Loading...")
#     f = open(fName, 'r')
#     data = json.load(f)
#     sSWS.set_val(data['SADWindowSize'])
#     sPFS.set_val(data['preFilterSize'])
#     sPFC.set_val(data['preFilterCap'])
#     sMDS.set_val(data['minDisparity'])
#     sNOD.set_val(data['numberOfDisparities'])
#     sTTH.set_val(data['textureThreshold'])
#     sUR.set_val(data['uniquenessRatio'])
#     sSR.set_val(data['speckleRange'])
#     sSPWS.set_val(data['speckleWindowSize'])
#     f.close()
#     buttonl.label.set_text("Load settings")
#     print('Parameters loaded from file ' + fName)
#     print('Redrawing depth map with loaded parameters...')
#     loading_settings = 0
#     update(0)
#     print('Done!')
#
#
# buttonl.on_clicked(load_map_settings)

# # Building Depth Map for the first time
# disparity = stereo_depth_map(rectified_pair)
#
# plt.subplot(1, 2, 2)
# dmObject = plt.imshow(disparity, aspect='equal', cmap='jet')
#
# # Draw interface for adjusting parameters
# print('Start interface creation (it takes up to 30 seconds)...')
#
# SWSaxe = plt.axes([0.15, 0.01, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
# PFSaxe = plt.axes([0.15, 0.05, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
# PFCaxe = plt.axes([0.15, 0.09, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
# MDSaxe = plt.axes([0.15, 0.13, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
# NODaxe = plt.axes([0.15, 0.17, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
# TTHaxe = plt.axes([0.15, 0.21, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
# URaxe = plt.axes([0.15, 0.25, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
# SRaxe = plt.axes([0.15, 0.29, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
# SPWSaxe = plt.axes([0.15, 0.33, 0.7, 0.025], facecolor=axcolor)  # stepX stepY width height
#
# sSWS = Slider(SWSaxe, 'SWS', 5.0, 255.0, valinit=5)
# sPFS = Slider(PFSaxe, 'PFS', 5.0, 255.0, valinit=5)
# sPFC = Slider(PFCaxe, 'PreFiltCap', 5.0, 63.0, valinit=29)
# sMDS = Slider(MDSaxe, 'MinDISP', -100.0, 100.0, valinit=-25)
# sNOD = Slider(NODaxe, 'NumOfDisp', 16.0, 256.0, valinit=128)
# sTTH = Slider(TTHaxe, 'TxtrThrshld', 0.0, 1000.0, valinit=100)
# sUR = Slider(URaxe, 'UnicRatio', 1.0, 20.0, valinit=10)
# sSR = Slider(SRaxe, 'SpcklRng', 0.0, 40.0, valinit=15)
# sSPWS = Slider(SPWSaxe, 'SpklWinSze', 0.0, 300.0, valinit=100)


# Update depth map parameters and redraw
# def update():
#     global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS
#     SWS = int(sSWS.val / 2) * 2 + 1  # convert to ODD
#     PFS = int(sPFS.val / 2) * 2 + 1
#     PFC = int(sPFC.val / 2) * 2 + 1
#     MDS = int(sMDS.val)
#     NOD = int(sNOD.val / 16) * 16
#     TTH = int(sTTH.val)
#     UR = int(sUR.val)
#     SR = int(sSR.val)
#     SPWS = int(sSPWS.val)
#     print('Rebuilding depth map')
#     disparity = stereo_depth_map(rectified_pair)
#     dmObject.set_data(disparity)
#     print('Redraw depth map')
#     plt.draw()


# Connect update actions to control elements
# sSWS.on_changed(update)
# sPFS.on_changed(update)
# sPFC.on_changed(update)
# sMDS.on_changed(update)
# sNOD.on_changed(update)
# sTTH.on_changed(update)
# sUR.on_changed(update)
# sSR.on_changed(update)
# sSPWS.on_changed(update)

# print('Show interface to user')
# plt.show()