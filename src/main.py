import cv2 as cv
import os
import ImageUtils
import VideoUtils
import numpy as np
import MultiImageViewer as view
import Viewers


if __name__ == "__main__":
    img = ImageUtils.readImage(ImageUtils.getDataPathWithFile("cat.png"))
    height, width = img.shape[:2]
    heightResize = height * 1.3
    widthResize = width * 1.3

    imgResizeArea = ImageUtils.resizeImage(img, int(widthResize), int(heightResize), interpolation=cv.INTER_AREA)
    imgResizeNearest = ImageUtils.resizeImage(img, int(widthResize), int(heightResize), interpolation=cv.INTER_NEAREST)
    imgResizeLinear = ImageUtils.resizeImage(img, int(widthResize), int(heightResize), interpolation=cv.INTER_LINEAR)
    imgResizeCubic = ImageUtils.resizeImage(img, int(widthResize), int(heightResize), interpolation=cv.INTER_CUBIC)
    imgResizeLanczos = ImageUtils.resizeImage(img, int(widthResize), int(heightResize), interpolation=cv.INTER_LANCZOS4)

    viewer = view.MultiImageViewer.from_images(img,imgResizeArea,imgResizeNearest,imgResizeLinear,imgResizeCubic,imgResizeLanczos, sync_view=False)
    viewer.run()

