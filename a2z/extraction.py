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


def extract(filelist, datasetFolder, output):
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
            instance = cv2.imread(datasetFolder + filename)
            
            #vetor de caracteristicas
            caract = list()

            #extrair caracteristicas aqui
            caract.extend( blackPixels(instance, 3, 3) )

            #salvo as caracteristicas no csv
            f.write(','.join([str(c) for c in caract]) + ',' + classe + '\n')
        #end for
    #end open
# end extraction

if __name__ == '__main__':
    extract('./test.txt', './NIST', 'teste.csv')