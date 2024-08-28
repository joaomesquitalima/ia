from sklearn.neural_network import MLPClassifier
import cv2 as cv
import os

import pickle
import numpy as np

diretorios = ['arduino/','led/','eu/','wall/']
dados = None
classes = np.array([])

num_diretorios = len(diretorios)

dados = []
classes = []

for i in range(num_diretorios):
    imagens = os.listdir(diretorios[i])

    for img in imagens:
        gray = cv.imread(os.path.join(diretorios[i], img), cv.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        gray = cv.resize(gray, (400, 400))
        dados.append(gray.flatten())
        classes.append(i)

dados = np.array(dados)
classes = np.array(classes)


redeNeural = MLPClassifier()

x , y = dados , classes

x = x / 255

num = 387

X_treino, X_teste = x[:num], x[num:]

y_treino, y_teste = y[:num], y[num:]

redeNeural.fit(X_treino, y_treino)

pickle.dump(redeNeural, open('modelo.sav', 'wb'))

print('Treino:', redeNeural.score(X_treino, y_treino))

print('Teste:', redeNeural.score(X_teste, y_teste))
