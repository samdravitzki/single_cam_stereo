import cv2
import numpy as np
import glob
from PIL import ExifTags
from PIL import Image
from matplotlib import pyplot
import numpy.linalg as la


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

        objpoints = []
        imgpoints = []
        # Prepare grid and the points to display
        objp = np.zeros((np.product(chessboard), 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)

        calibration_paths = glob.glob(input_directory)

        for i, image_path in enumerate(calibration_paths):
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print("Image loaded, Analysing {0}...".format(image_path))
            # find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray_image, chessboard, None)

            if ret == True:
                print("Chessboard detected!")
                # define criteria for subpixel accuracy
                critera = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
                # refine corner location based on criteria
                improved_corners = cv2.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), critera)
                objpoints.append(objp)
                imgpoints.append(improved_corners)
                image = cv2.drawChessboardCorners(image, chessboard, improved_corners, ret)
                cv2.imwrite("{0}draw-{1}.jpg".format(output_directory, i), image)
            else:
                print("Chessboard not detected!")
        # calibrate image based on the images
        ret, camera_matrix, distortion_coeff, rvecs, tvecs \
            = cv2.calibrateCamera(objpoints, imgpoints, gray_image.shape[::-1], None, None)

        self.ret = ret
        self.camera_matrix = camera_matrix
        self.distortion_coeff = distortion_coeff
        self.vecs = (rvecs, tvecs)

        # Get the focal-point form the mobile image meta data

        exif_image = Image.open(calibration_paths[0])

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

    def undistort_image(self, image):
        # Get the height and width (images must have equal size)
        image_height, image_width = image.shape[:2]

        # Calculate an optimal matrix based on the free scaling parameter
        # (A parameter that allows us to rescale the image and still keep a valid matrix)
        optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.distortion_coeff,
            (image_width, image_height), 1, (image_width, image_height))

        # Undistort the images with the optimal camera matrix
        image_undistorted = cv2.undistort(
            image,
            self.camera_matrix,
            self.distortion_coeff,
            newCameraMatrix=optimal_camera_matrix)

        # pyplot.imshow(image_undistorted, 'gray')
        # pyplot.show()

        x, y, w, h = roi
        image_undistorted = image_undistorted[y:y + h, x:x + w]

        # pyplot.imshow(image_undistorted, 'gray')
        # pyplot.show()

        return image_undistorted

    def rectify_stereo_pair(self, left, right):
        cv2.stereoRectify(cameraMatrix1=self.camera_matrix,
                          cameraMatrix2=self.camera_matrix,
                          distCoeffs1=self.distortion_coeff,
                          distCoeffs2=self.distortion_coeff,
                          imageSize=left.size[:2],
                          )

    def save_parameters(self):
        rvecs, tvecs = self.vecs
        np.save("./camera_parameters/ret", self.ret)
        np.save("./camera_parameters/K", self.camera_matrix)
        np.save("./camera_parameters/dist", self.distortion_coeff)
        np.save("./camera_parameters/rvecs", rvecs)
        np.save("./camera_parameters/tvecs", tvecs)
        np.save("./camera_parameters/focal_length", self.focal_length)

    def load_parameters(self):
        self.ret = np.load("./camera_parameters/ret.npy")
        self.camera_matrix = np.load("./camera_parameters/K.npy")
        self.distortion_coeff = np.load("./camera_parameters/dist.npy")
        rvecs = np.load("./camera_parameters/rvecs.npy")
        tvecs = np.load("./camera_parameters/tvecs.npy")
        self.vecs = (rvecs, tvecs)
        self.focal_length = np.load("./camera_parameters/focal_length.npy")


def rectify_stereo_pair_uncalibrated(imgL, imgR, calibrator):
    # imgL = calibrator.undistort_image(imgL)
    # imgR = calibrator.undistort_image(imgR)
    # findGoodPoints_opticalFlow(imgL, imgR)

    width, height = imgL.shape[:2]

    F, mask, left_points, right_points = findFundementalMatrix(imgL, imgR)

    # Select only inlier points
    left_points = left_points[mask.ravel() == 1]
    right_points = right_points[mask.ravel() == 1]

    linesL, linesR = calcualteEpilines(left_points, right_points, F)
    img5, img6 = drawlines(imgL.copy(), imgR.copy(), linesL, left_points, right_points)
    img3, img4 = drawlines(imgR.copy(), imgL.copy(), linesR, right_points, left_points)
    pyplot.subplot(121), pyplot.imshow(img5)
    pyplot.subplot(122), pyplot.imshow(img3)
    pyplot.show()

    # Rectify the images
    ret, h_left, h_right = cv2.stereoRectifyUncalibrated(left_points, right_points, F, (imgL.shape[1], imgL.shape[0]))

    # S = rectify_shearing(h_left, h_right, (imgL.shape[1], imgL.shape[0]))
    # h_left = S.dot(h_left)

    # Apply the rectification transforms to the images
    # camera_matrix = calibrator.camera_matrix
    # distortion = calibrator.distortion_coeff
    # imgsize = (imgL.shape[1], imgL.shape[0])
    # map1x, map1y, map2x, map2y = remap(camera_matrix, distortion, h_left, h_right, imgsize)
    #
    # rectified_left = cv2.remap(imgL, map1x, map1y,
    #                            interpolation=cv2.INTER_LINEAR)
    #
    # rectified_right = cv2.remap(imgR, map2x, map2y,
    #                             interpolation=cv2.INTER_LINEAR)

    rectified_left = cv2.warpPerspective(imgL, h_left, (height, width))
    rectified_right = cv2.warpPerspective(imgR, h_right, (height, width))

    ## DRAW RECALCULATED EPILINES ##
    # F, mask, left_points, right_points = findFundementalMatrix(rectified_left, rectified_right)
    #
    # # Select only inlier points
    # left_points = left_points[mask.ravel() == 1]
    # right_points = right_points[mask.ravel() == 1]
    #

    # linesL, linesR = calcualteEpilines(left_points, right_points, F)
    # rectified_left, img6 = drawlines(rectified_left.copy(), rectified_right.copy(), linesL, left_points, right_points)
    # rectified_right, img4 = drawlines(rectified_right.copy(), rectified_left.copy(), linesR, right_points, left_points)
    # pyplot.subplot(121), pyplot.imshow(img5)
    # pyplot.subplot(122), pyplot.imshow(img3)
    # pyplot.show()

    ## Display rectified images ##
    cv2.imshow('Left RECTIFIED', rectified_left)
    cv2.imshow('Right RECTIFIED', rectified_right)
    pyplot.show()
    cv2.waitKey(0)



    return rectified_left, rectified_right


def findFundementalMatrix(imgL, imgR):
    # # Initiate a SIFT detector for finding image feature points
    orb = cv2.ORB_create(2000)
    # # Find feature points in imgL
    keyPointsL, desL = orb.detectAndCompute(cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY), None)
    #
    # # Find corresponding feature points in imgR
    keyPointsR, desR = orb.detectAndCompute(cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY), None)

    left_points, right_points, good = findBestMatches(keyPointsL, keyPointsR, desL, desR)

    # matchedImg = None
    # matchedImg2 = cv2.drawMatches(imgL, keyPointsL, imgR, keyPointsR, good, flags=2, outImg=matchedImg)
    #
    # pyplot.imshow(matchedImg2), pyplot.show()
    #
    # cv2.waitKey(0)
    # Calculate fundemental matrix from the feature point pair
    F, mask = cv2.findFundamentalMat(left_points, right_points, cv2.FM_LMEDS)
    return F, mask, left_points, right_points

def calcualteEpilines(left_points, right_points, F):
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    linesL = cv2.computeCorrespondEpilines(right_points.reshape(-1, 1, 2), 2, F)
    linesL = linesL.reshape(-1, 3)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    linesR = cv2.computeCorrespondEpilines(left_points.reshape(-1, 1, 2), 1, F)
    linesR = linesR.reshape(-1, 3)

    return linesL, linesR


def drawlines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape[:2]
    # img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

def findBestMatches(keyPointsL, keyPointsR, desL, desR):
    # # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(desL, desR, k=2)
    # matches = sorted(matches, key=lambda x: x.distance)

    good = []
    ptsL = []
    ptsR = []

    for m, n in matches:
        if m.distance < 0.68 * n.distance:
            good.append(m)
            ptsR.append(keyPointsR[m.trainIdx].pt)
            ptsL.append(keyPointsL[m.queryIdx].pt)

    ptsL = np.float32(ptsL)
    ptsR = np.float32(ptsR)

    return ptsL, ptsR, good

def remap(camera_matrix, distortion, h_left, h_right, imgsize):
    r_left = la.inv(camera_matrix).dot(h_left).dot(camera_matrix)
    r_right = la.inv(camera_matrix).dot(h_right).dot(camera_matrix)
    height, width = imgsize

    optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion,
        (width, height), 1, (width, height))

    map1x, map1y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        r_left,
        optimal_camera_matrix,
        imgsize,
        cv2.CV_16SC2
    )

    map2x, map2y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        r_right,
        optimal_camera_matrix,
        imgsize,
        cv2.CV_16SC2
    )
    return map1x, map1y, map2x, map2y


def from_homg(x):
    """
    Transform homogeneous x to non-homogeneous coordinates
    If X is MxN, returns an (M-1)xN array that will contain nan when for
    columns where the last row was 0
    >>> from_homg(np.array([[1, 2, 3],
    ...                     [4, 5, 0]], dtype=float))
    array([[ 0.25,  0.4 ,   nan]])
    >>> from_homg(np.array([1, 5], dtype=float))
    array([ 0.2])
    >>> from_homg([1, 5, 0])
    array([ nan,  nan])
    >>> from_homg((1, 4, 0.5))end = time.time()
    array([ 2.,  8.])
    """
    if hasattr(x, 'shape') and len(x.shape) > 1:
        #valid = np.nonzero(x[-1,:])
        valid = x[-1,:] != 0
        result = np.empty((x.shape[0]-1, x.shape[1]), dtype=float)
        result[:,valid] = x[:-1,valid] / x[-1, valid]
        result[:,~valid] = np.nan
        return result
    else:
        if x[-1] == 0:
            result = np.empty(len(x)-1, dtype=float)
            result[:] = np.nan
            return result
        else:
            return np.array(x[:-1]) / x[-1]


def rectify_shearing(H1, H2, imsize):
    """Compute shearing transform than can be applied after the rectification
    transform to reduce distortion.
    See :
    http://scicomp.stackexchange.com/questions/2844/shearing-and-hartleys-rectification
    "Computing rectifying homographies for stereo vision" by Loop & Zhang
    """
    w = imsize[0]
    h = imsize[1]

    a = ((w-1)/2., 0., 1.)
    b = (w-1., (h-1.)/2., 1.)
    c = ((w-1.)/2., h-1., 1.)
    d = (0., (h-1.)/2., 1.)

    ap = from_homg(H1.dot(a))
    bp = from_homg(H1.dot(b))
    cp = from_homg(H1.dot(c))
    dp = from_homg(H1.dot(d))

    x = bp - dp
    y = cp - ap

    k1 = (h*h*x[1]*x[1] + w*w*y[1]*y[1]) / (h*w*(x[1]*y[0] - x[0]*y[1]))
    k2 = (h*h*x[0]*x[1] + w*w*y[0]*y[1]) / (h*w*(x[0]*y[1] - x[1]*y[0]))

    if k1 < 0:
        k1 *= -1
        k2 *= -1

    return np.array([[k1, k2, 0],
                     [0, 1, 0],
                     [0, 0, 1]], dtype=float)


def findGoodPoints_opticalFlow(imgL, imgR):
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Random colors
    color = np.random.randint(0, 25, (100, 3))
    maskR = np.zeros_like(imgR)

    left_points = cv2.goodFeaturesToTrack(grayL, mask=None, **feature_params)

    right_points, st, err = cv2.calcOpticalFlowPyrLK(grayL, grayR, left_points, None, **lk_params)

    # Select the good points
    good_right_points = right_points[st == 1]
    good_left_points = left_points[st == 1]

    # draw the tracks
    for i, (left, right) in enumerate(zip(good_left_points, good_right_points)):
        a, b = left.ravel()
        c, d = right.ravel()

        imgL = cv2.circle(imgL, (a, b), 4, color[i].tolist(), 2)
        imgR = cv2.circle(imgR, (c, d), 4, color[i].tolist(), 2)
        maskR = cv2.line(maskR, (a, b), (c, d), color[i].tolist(), 2)
        # imgR = cv2.circle(imgR, (a, b), 5, color[i].tolist(), -1)


    img = cv2.add(imgR, maskR)
    cv2.imshow('Left Image', imgL)
    cv2.imshow('Right Image', img)
    cv2.waitKey(0)