import cv2 as cv
import os
import ImageUtils

def videoFromWebcam():
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened():
        exit()
        
    while True:
        ret, frame = cap.read()
        
        if ret:
            cv.imshow('Webcam', frame)
            
        if cv.waitKey(1) == ord('q'):
            break
        
    cap.release()
    cv.destroyAllWindows()

def saveVideo(name, folder = 'output'):
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    output_path = ImageUtils.getDataPathWithFile(f'{folder}/{name}')

    out = cv.VideoWriter(output_path, 
                         cv.VideoWriter_fourcc(*'XVID'), 
                         20.0, 
                         (640, 480))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        out.write(frame)
        cv.imshow('Webcam', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()

def videoFromFile(name, folder = 'output'):
    video_path = ImageUtils.getDataPathWithFile(f'{folder}/{name}')
    cap = cv.VideoCapture(video_path)
    
    if not cap.isOpened():
        exit()
        
    while True:
        ret, frame = cap.read()
        
        if ret:
            cv.imshow('Video', frame)
            
        if cv.waitKey(30) == ord('q'):
            break
        
    cap.release()
    cv.destroyAllWindows()