import cv2 as cv
import os

webcam = cv.VideoCapture(0)


index_of_a= len(os.listdir('images/class_a'))
index_of_b = len(os.listdir('images/class_b'))
index_of_c = len(os.listdir('images/class_c'))

while True:
    _, img = webcam.read()

    cv.imshow("tela", img)

    tecla = cv.waitKey(1)

    if tecla == ord('a') and index_of_a <200:
        cv.imwrite(f'images/class_a/foto{index_of_a}.png',img)

        index_of_a+=1

    if tecla == ord('b') and index_of_b < 200:
        cv.imwrite(f'images/class_b/foto{index_of_b}.png',img)

        index_of_b+=1

    if tecla == ord('c') and index_of_c < 200:
        cv.imwrite(f'images/class_c/foto{index_of_c}.png',img)

        index_of_c+=1
