import cv2 as cv
import os
import matplotlib.pyplot as plt

def histogram(img):

    channels = img.shape[2] if len(img.shape) == 3 else 1

    colors = ['gray', 'blue', 'green', 'red']
    labels = ['Gray', 'Blue', 'Green', 'Red']

    plt.figure(figsize=(10, 5))

    if channels == 1:
        hist = cv.calcHist([img], [0], None, [256], [0, 256])
        plt.plot(hist, color='gray', label='Gray')

    else:
        for i in range(channels):
            hist = cv.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=colors[i], label=labels[i])

    plt.title('Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.legend()
    plt.grid()
    plt.show()