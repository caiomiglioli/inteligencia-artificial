import csv
import numpy as np
from termcolor import colored
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def openset(filename):
    #open
    data, classes = [], []
    with open(filename, 'r') as f:
        csvreader = csv.reader(f)
        next(csvreader)
        for row in csvreader:
            classes.append(row[-1])
            data.append(row[:-1])

    #normalize minmax
    data = normalize(np.array(data), axis=0, norm='max')

    return data, np.array(classes)
#end openset

def plotResult(true, pred, labels):
    cm = confusion_matrix(true, pred)
    print(
        "CONFUSION MATRIX --------------------------------------------------------------------------------\n"
        + '-\t' + colored('\t'.join(labels).upper(), 'red') 
    )
    for i in range(len(labels)):
        print(
            colored(labels[i].upper(), 'red')
            + '\t' + '\t'.join(  [str(n) if j!=i else colored(str(n),'green')      for j,n in enumerate(cm[i]) ]) 
        )    
    print(
        "\nACCURACY ----------------------------------------------------------------------------------------\n"
        + 'Acuracia: ' + f'{accuracy_score(true, pred) * 100:.2f}%'
    )
#end plotresult


if __name__ == '__main__':
    print('\n********************* OPEN ************************')
    x_train, y_train = openset('treino.csv')
    print('Treino: ', x_train.shape, y_train.shape)

    x_test, y_test = openset('teste.csv')
    print('Teste: ', x_test.shape, y_test.shape)

    # k-NN classifier
    print('\n********************* k-NN ************************')
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    plotResult(y_test, y_pred, labels=knn.classes_)