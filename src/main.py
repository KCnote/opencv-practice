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
    ip.histogram(img)
    
    viewer = view.MultiImageViewer.from_images(img,sync_view=False)
    viewer.run()

