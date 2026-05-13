import cv2 as cv
import os
import ImageUtils
import VideoUtils
import numpy as np
import MultiImageViewer as view
import Viewers
import ImageProcessing as ip


def saltPepperNoise(img, amount=0.02):
    noisy = img.copy()
    h, w = img.shape[:2]
    num_noise = int(h * w * amount)

    # Salt (white)
    y = np.random.randint(0, h, num_noise)
    x = np.random.randint(0, w, num_noise)

    noisy[y, x] = 255

    # Pepper (black)
    y = np.random.randint(0, h, num_noise)
    x = np.random.randint(0, w, num_noise)

    noisy[y, x] = 0

    return noisy

if __name__ == "__main__":
    img = ImageUtils.readImage(ImageUtils.getDataPathWithFile("cat.png"))
    img = saltPepperNoise(img, 0.05)
    imgGaussian = ip.filter3x3(img, ip.BlurType.GAUSSIAN_BLUR)
    imgUniform = ip.filter3x3(img, ip.BlurType.UNIFORM_BLUR)
    imgMedian = ip.filter3x3(img, ip.BlurType.MEDIAN_BLUR)
    imgBilateral = ip.filter3x3(img, ip.BlurType.BILATERAL_FILTER)
        
    viewer = view.MultiImageViewer.from_images(img, imgGaussian, imgUniform, imgMedian, imgBilateral, sync_view=False)
    viewer.run()

