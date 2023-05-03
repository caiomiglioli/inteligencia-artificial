"""
DONE - Normalizar os dados com Min-Max ou Z-score ;

DONE - Seu algoritmo deve avaliar o desempenho para diferentes valores de k {1,3,5,7,9,11,13,15,17,19} ;
DONE - Usar a distância Euclidiana ou Manhattan ;

entendi porra nenhuma (test) - Separar o conjunto de treinamento (aleatoriamente) em 25%, 50% e 100% dos dados de treinamento. Avaliar qual o impacto de usar mais e menos instâncias no conjunto de treinamento.

DONE - (testSet) - Usar o conjunto de características extraídas por vocês e apresentarem a taxa de acerto. Neste caso usar 100% das amostras de treinamento.
"""

import re
import numpy as np

from scipy.spatial import distance
from pandas import Series

from sklearn.preprocessing import normalize


class Knn:
    def __init__(self, filename):
        self.treino = self.__openFile(filename)
    #end constructor

    def testSet(self, filename, k, distance='Euclidian'):
        testset = self.__openFile(filename)
        # results = list()
        acertos = 0

        for teste in testset:
            dist = list()
            for treino in self.treino:
                if distance.lower() == 'manhattan':
                    dist.append(self.__distManhattan(teste,treino))
                else:
                    dist.append(self.__distEuclidian(teste,treino))

            dist.sort(key=lambda a : a['dist'])            
            p = Series([d['class'] for d in dist[:k]]).value_counts()

            # results.append({'result': p.keys()[0] == teste[-1], 'chute': p.keys()[0], 'classe': teste[-1]})
            if p.keys()[0] == teste[-1]:
                acertos += 1

        #exibir resultado
        print('===================== knn testset =====================')
        print(f'Configurações: teste={filename}, k={k}, dist={distance}')
        print(f'Taxa de acerto {acertos/testset.shape[0]*100:.1f}%')
        #return acertos/testset.shape[0]*100
    #end testset

    # ================================================================

    def __openFile(self, filename, classIndex=-1):
        with open(filename) as file:
            linhas = file.readlines()
            attrs = list()
            classes = list()

            for l in linhas:
                item = re.findall(r"[-+]?(?:\d*\.*\d+)", l)
                aux = [float(i) for i in item]
                attrs.append(aux[:classIndex:])
                classes.append(aux[classIndex])

            attrs = normalize(np.array(attrs), axis=0, norm='max')  #normalize os atributos
            return np.c_[attrs, classes]                            #concatena com a classe
    #end openfile

    def __distEuclidian(self, teste, treino):
        return { 'dist': distance.euclidean(teste[:-1], treino[:-1]), 'class': treino[-1] }
    #end euclidian

    def __distManhattan(self, teste, treino):
        return { 'dist': distance.cityblock(teste[:-1], treino[:-1]), 'class': treino[-1] }
    #end euclidian
#end knn

if __name__ == "__main__":
    knn = Knn('./digitos/treino_2x2.txt')
    knn.testSet('./digitos/teste_2x2.txt', k=3)