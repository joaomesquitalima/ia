import cv2 as cv
import numpy as np
from keras import models


modelo = models.load_model('MinhaSkyNet.h5')


webcam = cv.VideoCapture(0,cv.CAP_DSHOW)


nomes = [
     'eu',
     'Yoda',
     'calculadora'
]


while True:
    _ , img = webcam.read()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cnnInput= cv.resize(gray,(200,200))

    cnnInput = cnnInput.astype('float32')/255

    img2 = np.array([cnnInput])
    inferencia = modelo.predict(img2, verbose=0)

    previsao = inferencia[0]        
    classM = np.argmax(previsao)
    
    print(nomes[classM])


    cv.imshow('tela',img)

    cv.waitKey(1)