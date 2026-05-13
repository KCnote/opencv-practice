import cv2 as cv
import os
import ImageUtils
import VideoUtils
import MultiImageViewer as view


if __name__ == "__main__":
    img = ImageUtils.readImage(ImageUtils.getDataPathWithFile("cat.png"))
    img_b,img_g,img_r = cv.split(img)
    viewer = view.MultiImageViewer.from_images(img, img_b,img_g,img_r, sync_view=False)
    viewer.run()

