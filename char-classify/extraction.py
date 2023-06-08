import cv2
import math
import numpy as np

def blackPixels(image, x_qtde, y_qtde):
    pixels = list()
    h, w = image.shape[0]/x_qtde, image.shape[1]/y_qtde
    for i in range(x_qtde):
        for j in range(y_qtde):
            Xi, Xf = math.floor(i*h), math.floor((i+1)*h)
            Yi, Yf = math.floor(j*w), math.floor((j+1)*w)
            quadrante = image[Xi:Xf, Yi:Yf]            
            quadrante = np.average(quadrante, axis=-1).flatten()
            pixels.append(np.sum(quadrante < 128, axis=-1))
    return pixels
#end black pixels

def loadImage(filename):
    image = cv2.imread(filename)
    h, w = image.shape[0], image.shape[1]
    if h != w:
        margin = int(abs(h-w) / 2)
        mH, mW = (0, margin) if h>w else (margin, 0)
        image = cv2.copyMakeBorder(image, mH, mH, mW, mW, cv2.BORDER_CONSTANT, value=(255,255,255))
    image = cv2.resize(image, (60, 60))
    return image
#end loadimage


def extract(filelist, datasetFolder, output, maxInstances=None):
    balance = dict()

    #crio o  arquivo output (csv) e ja salvo o nome dos atributos
    with open(output, 'w') as f:
        csvLabels = ['bp1','bp2','bp3','bp4','bp5','bp6','bp7','bp8','bp9','classe']
        f.write(','.join(csvLabels) + '\n')

        #abro a lista de arquivos do conjunto (treino, teste, validação)
        with open(filelist) as fl:
            flist = [line.replace('\n', '') for line in fl.readlines()]
        
        #pra cada item da lista, eu abro a imagem, extraio as caracteristicas, e ja salvo no arquivo output
        for filename in flist:
            classe = filename.split('/')[1]

            #checar o balanceamento do conjunto
            if not balance.get(classe):
                balance[classe] = 0
            else:
                #quebrar o loop se a classe ja tiver com o maximo de instancias (force balance)
                if maxInstances and balance[classe] >= maxInstances:
                    continue
            balance[classe] += 1

            #load instance
            instance = loadImage(datasetFolder + filename)

            #vetor de caracteristicas
            caract = list()

            #extrair caracteristicas aqui
            caract.extend( blackPixels(instance, 3, 3) )

            #salvo as caracteristicas no csv
            f.write(','.join([str(c) for c in caract]) + ',' + classe + '\n')
        #end for
    #end open
    print('Balanceamento:', balance)
# end extraction

if __name__ == '__main__':
    extract('./datasets/NIST_Train_Upper.txt', './NIST', output='./output/treino.csv', maxInstances=500)
    extract('./datasets/NIST_Test_Upper.txt', './NIST', output='./output/teste.csv')
    extract('./datasets/NIST_Valid_Upper.txt', './NIST', output='./output/validacao.csv')