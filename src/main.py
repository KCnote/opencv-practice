import cv2 as cv
import st_image

if __name__ == "__main__":
    img = st_image.readImage(st_image.getImagePath("cat.png"))
    st_image.showImage("Cat", img)
