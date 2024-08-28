import pickle
import cv2 as cv
import os
import numpy as np
webcam = cv.VideoCapture(0, cv.CAP_DSHOW)

modelo = pickle.load(open("modelo.sav", "rb"))

while True:
    cap , img = webcam.read()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    gray = cv.resize(gray,(400,400))
    
    dados = gray.flatten()
    dados = np.array([dados])

   

    previsoes = modelo.predict(dados)
    
    if previsoes[0] == 0:
        cv.putText(img,"ARDUINO",(10,30),cv.FONT_HERSHEY_SIMPLEX,1,255)
        print('arduino')
        
         
    if previsoes[0] == 1 :
        print('led')
        cv.putText(img,"LED",(10,30),cv.FONT_HERSHEY_SIMPLEX,1,255)

    if previsoes[0] == 2:
        print('eu')
        cv.putText(img,"eu",(10,30),cv.FONT_HERSHEY_SIMPLEX,1,255)

    if previsoes[0] == 3:
        print('parede')
        
        cv.putText(img,"PAREDE",(10,30),cv.FONT_HERSHEY_SIMPLEX,1,255)



    cv.imshow("Tela", img)


    cv.waitKey(1)
