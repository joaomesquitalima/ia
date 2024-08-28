
import cv2 as cv
import os
minhaWebcam = cv.VideoCapture(0,cv.CAP_DSHOW)
capturou, img = minhaWebcam.read()


v = 0
arquivo = f'C:/Users/VisaoComput/Desktop/joao/pythonzin/inteligencia_falsa/ex_1/wall/wall{v}.jpg'

while True:

    if capturou:
        capturou, img = minhaWebcam.read()
        cv.imshow("Webcan",img)


        tecla = cv.waitKey(1)
        if tecla == ord('a'):
            break
        if tecla == ord('l'):
            if os.path.exists(arquivo):
                v+=1
            cv.imwrite(f'C:/Users/VisaoComput/Desktop/joao/pythonzin/inteligencia_falsa/ex_1/wall/wall{v}.jpg',img)


