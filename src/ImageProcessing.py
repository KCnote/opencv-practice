import cv2 as cv
import os
import matplotlib.pyplot as plt

import cv2 as cv
import numpy as np
from enum import Enum


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


def morphology(
    img,
    morphology_type,
    ksize=3,
    iterations=1
):

    if ksize % 2 == 0:
        raise ValueError("ksize must be odd")

    kernel = cv.getStructuringElement(
        cv.MORPH_RECT,
        (ksize, ksize)
    )

    if morphology_type == MorphologyType.EROSION:

        return cv.erode(
            img,
            kernel,
            iterations=iterations
        )

    elif morphology_type == MorphologyType.DILATION:

        return cv.dilate(
            img,
            kernel,
            iterations=iterations
        )

    elif morphology_type == MorphologyType.OPENING:

        return cv.morphologyEx(
            img,
            cv.MORPH_OPEN,
            kernel
        )

    elif morphology_type == MorphologyType.CLOSING:

        return cv.morphologyEx(
            img,
            cv.MORPH_CLOSE,
            kernel
        )

    elif morphology_type == MorphologyType.MORPH_GRADIENT:

        return cv.morphologyEx(
            img,
            cv.MORPH_GRADIENT,
            kernel
        )

    elif morphology_type == MorphologyType.TOP_HAT:

        return cv.morphologyEx(
            img,
            cv.MORPH_TOPHAT,
            kernel
        )

    elif morphology_type == MorphologyType.BLACK_HAT:

        return cv.morphologyEx(
            img,
            cv.MORPH_BLACKHAT,
            kernel
        )

    else:

        raise ValueError("Invalid morphology type")

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