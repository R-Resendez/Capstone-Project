import cv2
import numpy as np




#Webcam Setup and Parameters

web_cam = cv2.VideoCapture(0)#defaults to first webcam detected
web_cam.set(3,640)#set the capture window width
web_cam.set(4,480)#set the capture window height



while True:
    success, img = web_cam.read()
    cv2.imshow("Webcam Live Feed",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
