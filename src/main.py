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
    imgErode = ip.morphology3x3(img, ip.MorphologyType.EROSION)
    imgDilate = ip.morphology3x3(img, ip.MorphologyType.DILATION)
    imgOpen = ip.morphology3x3(img, ip.MorphologyType.OPENING)
    imgClose = ip.morphology3x3(img, ip.MorphologyType.CLOSING)
    imgGradient = ip.morphology3x3(img, ip.MorphologyType.MORPH_GRADIENT)
    imgTopHat = ip.morphology3x3(img, ip.MorphologyType.TOP_HAT)
    imgBlackHat = ip.morphology3x3(img, ip.MorphologyType.BLACK_HAT)

    viewer = view.MultiImageViewer.from_images(img, imgErode, imgDilate, imgOpen, imgClose, imgGradient, imgTopHat, imgBlackHat, sync_view=False)
    viewer.run()

