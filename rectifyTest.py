import cv2
import numpy as np

n_boards = 5
board_w = 7
board_h = 4
square_sz = float(10.0)







def hartleyRectify(points1, points2, imgSize, M1, M2, D1, D2, F):
    # F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, 3, 0.99)
    # print 'mask\n', mask
    retval, H1, H2 = cv2.stereoRectifyUncalibrated(
        points1, points2, F, imgSize)
    retval, M1i = cv2.invert(M1);
    retval, M2i = cv2.invert(M2)
    R1, R2 = np.dot(np.dot(M1i, H1), M1), np.dot(np.dot(M2i, H2), M2)
    map1x, map1y = cv2.initUndistortRectifyMap(M1, D1, R1, M1, imgSize, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(M2, D2, R2, M2, imgSize, cv2.CV_32FC1)
    return (map1x, map1y, map2x, map2y), F





def stereo_rectify(img1fn, img2fn, mapfn, qfn):
    urmaps = np.load(mapfn)
    Q = np.load(qfn)
    img1 = cv2.imread(img1fn, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2fn, cv2.IMREAD_COLOR)
    imgL = cv2.remap(img1, urmaps[0], urmaps[1], cv2.INTER_LINEAR)
    imgR = cv2.remap(img2, urmaps[2], urmaps[3], cv2.INTER_LINEAR)
    cv2.imshow('Image L', imgL);
    cv2.imshow('Image R', imgR)
    cv2.waitKey(0)
    return imgL, imgR, Q

