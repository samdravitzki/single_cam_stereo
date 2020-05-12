'''
Simple example of stereo image matching and point cloud generation.
Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

'''
Simple example of stereo image matching and point cloud generation.
Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

import numpy as np
import cv2 as cv
from Calibrator import Calibrator


def downsample_image(image, reduce_factor):
    for i in range(0, reduce_factor):
        # If the image is grayscale get the size differently
        if len(image.shape) > 2:
            row, col = image.shape[:2]
        else:
            row, col = image.shape

        # Scale the image down by half each time
        image = cv.pyrDown(image, dstsize=(col // 2, row // 2))

    return image


def main():
    print('loading images...')
    # imgL = downsample_image(cv.imread("./input_images/aloeL.jpg"), 1)  # downscale images for faster processing
    # imgR = downsample_image(cv.imread("./input_images/aloeR.jpg"), 1)
    imgL = downsample_image(cv.imread("./input_images/left3.jpg"), 3)  # downscale images for faster processing
    imgR = downsample_image(cv.imread("./input_images/right3.jpg"), 3)
    calibrator = Calibrator(6, 9)
    calibrator.load_parameters()
    imgL = calibrator.undistort_image(imgL)
    imgR = calibrator.undistort_image(imgR)

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = -1
    num_disp = 112 - min_disp
    left_matcher = cv.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=16,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    print('computing disparity...')
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)

    # Use wls filter to get a hole free image
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0

    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    print("computing and filtering disparity...")
    disparity_left = left_matcher.compute(imgL, imgR)
    disparity_right = right_matcher.compute(imgR, imgL)
    disparity_left = np.int16(disparity_left)
    disparity_right = np.int16(disparity_right)
    filtered_img = wls_filter.filter(disparity_left, imgL, None, disparity_right)
    # Normalise filtered image
    norm_filtered_img = ((filtered_img.astype(np.float32) / 16.0) - min_disp) / num_disp

    cv.imshow('left', imgL)
    cv.imshow('right', imgR)
    cv.imshow('disparity', norm_filtered_img)
    cv.waitKey()

    print('Done')

# TODO implement the filtering into the Tuners
if __name__ == '__main__':
    main()


#
# import numpy as np
# import cv2 as cv
#
# def downsample_image(image, reduce_factor):
#     for i in range(0, reduce_factor):
#         # If the image is grayscale get the size differently
#         if len(image.shape) > 2:
#             row, col = image.shape[:2]
#         else:
#             row, col = image.shape
#
#         # Scale the image down by half each time
#         image = cv.pyrDown(image, dstsize=(col // 2, row // 2))
#
#     return image
#
# def main():
#     print('loading images...')
#     imgL = downsample_image(cv.imread("./input_images/aloeL.jpg"), 1)  # downscale images for faster processing
#     imgR = downsample_image(cv.imread("./input_images/aloeR.jpg"), 1)
#     # imgL = downsample_image(cv.imread("./input_images/left3.jpg"), 3)  # downscale images for faster processing
#     # imgR = downsample_image(cv.imread("./input_images/right3.jpg"), 3)
#
#     # disparity range is tuned for 'aloe' image pair
#     window_size = 3
#     min_disp = -1
#     num_disp = 112-min_disp
#     stereo = cv.StereoSGBM_create(
#         minDisparity = min_disp,
#         numDisparities = num_disp,
#         blockSize = 16,
#         P1 = 8*3*window_size**2,
#         P2 = 32*3*window_size**2,
#         disp12MaxDiff = 1,
#         uniquenessRatio = 10,
#         speckleWindowSize = 100,
#         speckleRange = 32
#     )
#
#     print('computing disparity...')
#     disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
#
#     cv.imshow('left', imgL)
#     cv.imshow('disparity', (disp-min_disp)/num_disp)
#     cv.waitKey()
#
#     print('Done')
#
#
# if __name__ == '__main__':
#     main()