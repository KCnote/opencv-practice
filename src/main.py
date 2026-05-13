import cv2 as cv
import os
import ImageUtils
import VideoUtils
import numpy as np
import MultiImageViewer as view
import Viewers
import ImageProcessing as ip

if __name__ == "__main__":
    #img = ImageUtils.readImage(ImageUtils.getDataPathWithFile("cat.png"))
    
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        exit()
    
    ret, img = cap.read()
    
    viewer = view.MultiImageViewer.from_images(img, img, sync_view=False)
    
    while True:
        ret, img = cap.read()
        
        if not ret:
            break

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        imgF32 = np.float32(gray)
        harrisInfo = cv.cornerHarris(imgF32, 2, 3, 0.04)
        harris = cv.dilate(harrisInfo, None)

        img[harris > 0.01 * harris.max()] = [0, 0, 255]
        harrisMap = harris / harris.max() * 255

        if ret:
            viewer.update_images(img, harrisMap)
            viewer.draw()
        
        key = cv.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break