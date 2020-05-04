'''
Simple example of stereo image matching and point cloud generation.
Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''


import numpy as np
import cv2 as cv

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

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv.StereoSGBM_create(
        minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    cv.imshow('left', imgL)
    cv.imshow('disparity', (disp-min_disp)/num_disp)
    cv.waitKey()

    print('Done')


if __name__ == '__main__':
    main()