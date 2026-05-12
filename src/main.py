import cv2 as cv
import os
import ImageUtils
import VideoUtils
import MultiImageViewer as view

if __name__ == "__main__":
    VideoUtils.videoFromWebcam()
    img = ImageUtils.readImage(ImageUtils.getDataPathWithFile("cat.png"))
    viewer = view.MultiImageViewer("data/cat.png", "data/cat.png", sync_view=False)
    viewer.run()

