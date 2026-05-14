import cv2 as cv
import os
import matplotlib.pyplot as plt

import cv2 as cv
import numpy as np
from enum import Enum
import ImageUtils


class KernelType(Enum):
    GAUSSIAN_BLUR = 0
    UNIFORM_BLUR = 1
    SHARPEN = 2
    LAPLACIAN = 3
    SOBEL_X = 4
    SOBEL_Y = 5
    EMBOSS = 6

def convolution3x3(img, kernel_type):
    if kernel_type == KernelType.GAUSSIAN_BLUR:
        kernel_1d = cv.getGaussianKernel(3, 0)
        kernel = kernel_1d @ kernel_1d.T
    elif kernel_type == KernelType.UNIFORM_BLUR:
        kernel = np.ones((3, 3), dtype=np.float32)
        kernel /= 9
    elif kernel_type == KernelType.SHARPEN:
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
    elif kernel_type == KernelType.LAPLACIAN:
        kernel = np.array([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=np.float32)
    elif kernel_type == KernelType.SOBEL_X:
        kernel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32)
    elif kernel_type == KernelType.SOBEL_Y:
        kernel = np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=np.float32)
    elif kernel_type == KernelType.EMBOSS:
        kernel = np.array([
            [-2, -1, 0],
            [-1,  1, 1],
            [ 0,  1, 2]
        ], dtype=np.float32)

    else:
        raise ValueError("Invalid kernel type")

    return cv.filter2D(img, -1, kernel)

import cv2 as cv
from enum import Enum


class BlurType(Enum):
    UNIFORM_BLUR = 0
    GAUSSIAN_BLUR = 1
    MEDIAN_BLUR = 2
    BILATERAL_FILTER = 3


def filter3x3(img, blur_type):
    if blur_type == BlurType.UNIFORM_BLUR:
        return cv.blur(img, (3, 3))
    elif blur_type == BlurType.GAUSSIAN_BLUR:
        return cv.GaussianBlur(img, (3, 3), 0)
    elif blur_type == BlurType.MEDIAN_BLUR:
        return cv.medianBlur(img, 3)
    elif blur_type == BlurType.BILATERAL_FILTER:
        return cv.bilateralFilter(img, 9, 75, 75)

class MorphologyType(Enum):
    EROSION = 0
    DILATION = 1
    OPENING = 2
    CLOSING = 3
    MORPH_GRADIENT = 4
    TOP_HAT = 5
    BLACK_HAT = 6


def morphology3x3(img, morphology_type):

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    if morphology_type == MorphologyType.EROSION:
        return cv.erode(img, kernel, iterations=1)
    elif morphology_type == MorphologyType.DILATION:
        return cv.dilate(img, kernel, iterations=1)
    elif morphology_type == MorphologyType.OPENING:
        return cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    elif morphology_type == MorphologyType.CLOSING:
        return cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    elif morphology_type == MorphologyType.MORPH_GRADIENT:
        return cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
    elif morphology_type == MorphologyType.TOP_HAT:
        return cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
    elif morphology_type == MorphologyType.BLACK_HAT:
        return cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

class ThresholdType(Enum):
    BINARY = 0
    BINARY_INV = 1
    TRUNC = 2
    TOZERO = 3
    TOZERO_INV = 4

    OTSU = 5
    ADAPTIVE_MEAN = 6
    ADAPTIVE_GAUSSIAN = 7


def thresholding(img, threshold_type, thresh=127, max_value=255, block_size=11, c=2):

    # Adaptive threshold requires grayscale
    if len(img.shape) == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img

    if threshold_type == ThresholdType.BINARY:
        _, result = cv.threshold(gray, thresh, max_value, cv.THRESH_BINARY)
    elif threshold_type == ThresholdType.BINARY_INV:
        _, result = cv.threshold(gray, thresh, max_value, cv.THRESH_BINARY_INV)
    elif threshold_type == ThresholdType.TRUNC:
        _, result = cv.threshold(gray, thresh, max_value, cv.THRESH_TRUNC)
    elif threshold_type == ThresholdType.TOZERO:
        _, result = cv.threshold(gray, thresh, max_value, cv.THRESH_TOZERO)
    elif threshold_type == ThresholdType.TOZERO_INV:
        _, result = cv.threshold(gray, thresh, max_value, cv.THRESH_TOZERO_INV)
    elif threshold_type == ThresholdType.OTSU:
        _, result = cv.threshold(gray, 0, max_value, cv.THRESH_BINARY | cv.THRESH_OTSU)
    elif threshold_type == ThresholdType.ADAPTIVE_MEAN:
        result = cv.adaptiveThreshold(gray, max_value, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block_size, c)
    elif threshold_type == ThresholdType.ADAPTIVE_GAUSSIAN:
        result = cv.adaptiveThreshold(gray, max_value, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, c)

    return result



def histogram(img):

    channels = img.shape[2] if len(img.shape) == 3 else 1

    colors = ['gray', 'blue', 'green', 'red']
    labels = ['Gray', 'Blue', 'Green', 'Red']

    plt.figure(figsize=(10, 5))

    if channels == 1:
        hist = cv.calcHist([img], [0], None, [256], [0, 256])
        plt.plot(hist, color='gray', label='Gray')

    else:
        for i in range(channels):
            hist = cv.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=colors[i], label=labels[i])

    plt.title('Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.legend()
    plt.grid()
    plt.show()
    
def HoughLines(img, rho=1, theta=np.pi/180, threshold=100):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLines(edges, rho, theta, threshold)

    return lines


def calibrate_camera_from_images(img_path):
    checkerboard_candidates = [(9, 6),(8, 6),(7, 6),(6, 5),(5, 4),(8, 5),(7, 5),(10, 7)]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,30,0.001)

    objpoints = []
    imgpoints = []
    img_size = None
    selected_checkerboard = None

    for i in range(1, 7):
        img = cv.imread(ImageUtils.getDataPathWithFile(f"{img_path}\\cab{i}.png"))

        if img is None:
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        found_any = False

        for checkerboard in checkerboard_candidates:
            found, corners = cv.findChessboardCorners(gray,checkerboard,cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)

            if not found:
                continue

            print(f"Detected corners: cab{i}.png, pattern={checkerboard}")

            objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:checkerboard[0],0:checkerboard[1]].T.reshape(-1, 2)
            corners_refined = cv.cornerSubPix(gray,corners,(11, 11),(-1, -1),criteria)

            objpoints.append(objp)
            imgpoints.append(corners_refined)

            debug = img.copy()
            cv.drawChessboardCorners(debug,checkerboard,corners_refined,found)
            cv.imshow("Detected Corners", debug)
            cv.waitKey(300)

            selected_checkerboard = checkerboard
            found_any = True
            break

        if not found_any:
            print(f"Corners not found: cab{i}.png")

    cv.destroyAllWindows()

    if len(objpoints) == 0:
        raise RuntimeError("No chessboard corners detected. Check inner-corner count, blur, lighting, or whether the full board is visible.")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(objpoints,imgpoints,img_size,None,None)

    print("Selected checkerboard:", selected_checkerboard)
    print("Calibration RMS error:", ret)
    print("Camera Matrix:")
    print(camera_matrix)
    print("Distortion Coefficients:")
    print(dist_coeffs)

    return camera_matrix, dist_coeffs

def undistort_image(img, camera_matrix, dist_coeffs, new_camera_matrix=None):
    if img is None:
        return

    h, w = img.shape[:2]
    
    if new_camera_matrix is None:
        new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix,dist_coeffs,(w, h),alpha=1,newImgSize=(w, h))

    undistorted = cv.undistort(img,camera_matrix,dist_coeffs,None,new_camera_matrix)
    return undistorted
