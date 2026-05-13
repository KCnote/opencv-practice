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
    
def convertColorSpace(img, colorCode = cv.COLOR_BGR2GRAY):
    return cv.cvtColor(img, colorCode)

def resizeImage(img, widthResize, heightResize, interpolation=cv.INTER_AREA):
    return cv.resize(img, (int(widthResize), int(heightResize)), interpolation=interpolation)
