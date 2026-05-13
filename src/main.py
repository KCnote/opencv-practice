import cv2 as cv
import os
import ImageUtils
import VideoUtils
import numpy as np
import MultiImageViewer as view
import Viewers
import ImageProcessing as ip

if __name__ == "__main__":
    img = ImageUtils.readImage(ImageUtils.getDataPathWithFile("road.png"))
    imgGauss = ip.convolution3x3(img, ip.KernelType.GAUSSIAN_BLUR)
    lines = ip.HoughLines(imgGauss, 1, np.pi/180, 200)

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            slope = (y2 - y1) / (x2 - x1)
            
            if abs(slope) < 0.3:
                continue
            
            cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    viewer = view.MultiImageViewer.from_images(img, sync_view=False)
    viewer.run()