"""
DONE - Normalizar os dados com Min-Max ou Z-score ;

DONE - Seu algoritmo deve avaliar o desempenho para diferentes valores de k {1,3,5,7,9,11,13,15,17,19} ;
DONE - Usar a distância Euclidiana ou Manhattan ;

DONE - Separar o conjunto de treinamento (aleatoriamente) em 25%, 50% e 100% dos dados de treinamento. 
TODO - Avaliar qual o impacto de usar mais e menos instâncias no conjunto de treinamento.

    - Ao utilizar apenas 25% do conjunto de treinamento, pode ser observada uma redução considerável da porcentagem de acertos- caindo
    mais de 10% em alguns casos - já ao utilizar 50% do conjunto o resultado se manteve em torno de 80%, que é próximo ao resultado com
    o conjunto todo (83.3%). Com isso, afirma-se que é possível utilizar menos dados e manter uma boa taxa de acertos, dessa forma o 
    treinamento é otimizando.

DONE - (testSet) - Usar o conjunto de características extraídas por vocês e apresentarem a taxa de acerto. Neste caso usar 100% das amostras de treinamento.
"""

import re
import numpy as np
import random

from scipy.spatial import distance
from pandas import Series

from sklearn.preprocessing import normalize


class Knn:
    def __init__(self, filename, porcentagem_conjunto=1):
        self.treino = self.__openFile(filename, tipo_arquivo='treino', porcentagem_conjunto=porcentagem_conjunto)
    #end constructor

    def testSet(self, filename, k, distance='Euclidian'):
        testset = self.__openFile(filename, tipo_arquivo='teste')
        # results = list()
        acertos = 0

        print('Realizando testes...')
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
        print(f'Taxa de acerto {acertos/testset.shape[0]*100:.1f}%\n')

        with open('resultados.txt', 'a') as file:
            file.write(f',{acertos/testset.shape[0]*100:.1f}\n')

        #return acertos/testset.shape[0]*100
    #end testset

    # ================================================================

    def __openFile(self, filename, tipo_arquivo, porcentagem_conjunto=1, classIndex=-1):
        #print(f'abrindo arquivo {filename.split("/")[-1]}...\n')
        try:
            with open(filename) as file:
                linhas = file.readlines()

                if (tipo_arquivo == "treino" and porcentagem_conjunto != 1):
                    quantidade_linhas_total = len(linhas)
                    quantidade_linhas_usadas = int(quantidade_linhas_total * porcentagem_conjunto)

                    indices_linhas_usadas = random.sample(range(quantidade_linhas_total), quantidade_linhas_usadas)
                    linhas = [linhas[i] for i in indices_linhas_usadas]

                attrs = list()
                classes = list()

                for linha in linhas:
                    item = re.findall(r"[-+]?(?:\d*\.*\d+)", linha)
                    aux = [float(i) for i in item]
                    attrs.append(aux[:classIndex:])
                    classes.append(aux[classIndex])

                attrs = normalize(np.array(attrs), axis=0, norm='max')  #normalize os atributos
                #print('arquivo aberto!')
                return np.c_[attrs, classes]                            #concatena com a classe
            
        except:
            print(f'Erro ao abrir arquivo {filename.split("/")[-1]}')

    #end openfile

    def __distEuclidian(self, teste, treino):
        return { 'dist': distance.euclidean(teste[:-1], treino[:-1]), 'class': treino[-1] }
    #end euclidian

    def __distManhattan(self, teste, treino):
        return { 'dist': distance.cityblock(teste[:-1], treino[:-1]), 'class': treino[-1] }
    #end euclidian
#end knn

if __name__ == "__main__":
    #knn = Knn('./digitos/treino_2x2.txt', porcentagem_conjunto=1)
    #knn.testSet('./digitos/teste_2x2.txt', k=3)

    with open('resultados.txt', 'w') as file:
        file.write('porcentagem_usada_treino,resultado\n')

    for i in range(0, 3):
        for j in range(1, 10):
            print(i, j)
            pct_cnj = 0.25
            if i == 1: 
                pct_cnj = 0.5

            if i == 2: 
                pct_cnj = 1

            with open('resultados.txt', 'a') as file:
                file.write(f'{pct_cnj}')
                

            knn = Knn('./digitos/treino_2x2.txt', porcentagem_conjunto=pct_cnj)
            knn.testSet('./digitos/teste_2x2.txt', k=3)

            if i == 2:
                break

