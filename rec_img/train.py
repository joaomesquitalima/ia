import numpy as np
import os
import cv2 as cv
import tensorflow as tf
from tf_keras import Sequential
from sklearn.model_selection import train_test_split
from tf_keras.utils import to_categorical 
from tf_keras.layers import Dense , Conv2D, MaxPooling2D , Dropout, Flatten


pastas = ['images/class_a/','images/class_b/','images/class_c/']

dados = []
classes = []

qtd_pastas = len(pastas)

for pasta in range(qtd_pastas):
    imagens = os.listdir(pastas[pasta])

    for imagem in imagens:
        gray = cv.imread(pastas[pasta] + imagem, cv.IMREAD_GRAYSCALE)
        gray = cv.resize(gray,(200,200))
        if gray is None:
            continue


        dados.append(gray)
        classes.append(pasta)

x , y = np.array(dados) , np.array(classes)

X_treino, X_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)

X_treino = X_treino.astype("float32") / 255

y_treino = to_categorical(y_treino)
y_teste = to_categorical(y_teste)


modelo = Sequential()

modelo = Sequential()
modelo.add(Conv2D(50, (3, 3), activation='relu', input_shape=(200, 200, 1)))
modelo.add(MaxPooling2D())
modelo.add(Dropout(0.2))
modelo.add(Flatten())
modelo.add(Dense(200, activation='relu'))
modelo.add(Dropout(0.3))
modelo.add(Dense(3, activation='softmax'))


modelo.summary()

modelo.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])



modelo.fit(
    X_treino, 
    y_treino, 
    batch_size=32, 
    epochs=50, 
    verbose=1, 
    validation_split=0.2,

)

modelo.save("MinhaSkyNet.h5")

