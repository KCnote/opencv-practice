import cv2 as cv
import os
import ImageUtils
import VideoUtils
import numpy as np
import MultiImageViewer as view
import Viewers
import ImageProcessing as ip

if __name__ == "__main__":
    img = ImageUtils.readImage(ImageUtils.getDataPathWithFile("cat.png"))
    imgThresholdBinary = ip.thresholding(img, ip.ThresholdType.BINARY)
    imgThresholdBinaryInv = ip.thresholding(img, ip.ThresholdType.BINARY_INV)
    imgThresholdTrunc = ip.thresholding(img, ip.ThresholdType.TRUNC)
    imgThresholdToZero = ip.thresholding(img, ip.ThresholdType.TOZERO)
    imgThresholdToZeroInv = ip.thresholding(img, ip.ThresholdType.TOZERO_INV)
    imgThresholdAdaptiveMean = ip.thresholding(img, ip.ThresholdType.ADAPTIVE_MEAN, block_size=11, c=2)
    imgThresholdAdaptiveGaussian = ip.thresholding(img, ip.ThresholdType.ADAPTIVE_GAUSSIAN, block_size=11, c=2)
    imgThresholdOtsu = ip.thresholding(img, ip.ThresholdType.OTSU)

    viewer = view.MultiImageViewer.from_images(img, imgThresholdBinary, imgThresholdBinaryInv, imgThresholdTrunc, imgThresholdToZero, imgThresholdToZeroInv, imgThresholdAdaptiveMean, imgThresholdAdaptiveGaussian, imgThresholdOtsu, sync_view=False)
    viewer.run()

