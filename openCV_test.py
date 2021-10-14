import cv2
import numpy as np


faceCascade = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")

#Webcam Setup and Parameters

web_cam = cv2.VideoCapture(0)#defaults to first webcam detected
web_cam.set(3,640)#set the capture window width
web_cam.set(4,480)#set the capture window height



while True:
    success, img = web_cam.read()

    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(imgGray,1.1,4)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)#Important function for visual debugging
    cv2.imshow("Webcam Live Feed",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
