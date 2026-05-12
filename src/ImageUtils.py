import cv2 as cv
import os

def getDataPath():
    strRoot = os.getcwd()
    strDir = os.path.join(strRoot, "data")
    return strDir

def getDataPathWithFile(path):
    strImage = os.path.join(getDataPath(), path)
    return strImage

def readImage(path):
    img = cv.imread(path)
    return img

def showImage(title, img):
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def writeImage(path, img):
    cv.imwrite(path, img)