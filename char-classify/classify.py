import csv
import numpy as np
from termcolor import colored

from sklearn import svm, tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

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
    def tab(value):
        TABSIZE = 5
        spaces = TABSIZE - len(str(value))
        return str(value) + (' ' * spaces)
    
    cm = confusion_matrix(true, pred)
    print("CONFUSION MATRIX --------------------------------------------------------------------------------\n")
    print(colored( tab('-') + ''.join([tab(l.upper()) for l in labels]) , 'red', 'on_black'))
    
    for i in range(len(labels)):
        print(
            colored( tab(labels[i].upper()), 'red', 'on_black')
            + ''.join([     tab(n)     if j!=i else     colored(tab(n),'green')      for j,n in enumerate(cm[i])     ]) 
        )
        
    print('\nACCURACY ----------------------------------------------------------------------------------------')
    print('Acuracia: ' + f'{accuracy_score(true, pred) * 100:.2f}%\n')
#end plotresult

def main():
    print('\n********************* OPEN ************************')
    x_train, y_train = openset('./output/treino.csv')
    print('Treino: ', x_train.shape, y_train.shape)
    x_test, y_test = openset('./output/teste.csv')
    print('Teste: ', x_test.shape, y_test.shape)

    # ------------------------------------------------------------
    # Decision Tree classifier
    print('\n********************** DT *************************')
    dt = tree.DecisionTreeClassifier()
    dt = dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    plotResult(y_test, y_pred, labels=dt.classes_)

    # ------------------------------------------------------------
    # k-NN classifier
    print('\n********************* k-NN ************************')
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    plotResult(y_test, y_pred, labels=knn.classes_)

    # ------------------------------------------------------------
    # Random Forest classifier
    print('\n***************** Random Forest *******************')
    rf = RandomForestClassifier(n_estimators=1000, max_depth=30, random_state=1)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    plotResult(y_test, y_pred, labels=rf.classes_)

    # ------------------------------------------------------------
    # Multi Layer Perceptron classifier
    print('\n********************** MLP ************************')
    mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100, 100), random_state=5, max_iter=500)
    mlp.fit(x_train, y_train)
    y_pred = mlp.predict(x_test)
    plotResult(y_test, y_pred, labels=mlp.classes_)

    # ------------------------------------------------------------
    # SVM classifier
    print('\n********************* SVM ************************')
    # pipeline -> instancia o classificador, gerando probabilidades
    srv = svm.SVC(probability=True, kernel='rbf')
    ss = StandardScaler()
    pipeline = Pipeline([ ('scaler', ss), ('svm', srv) ])
    
    # gridsearch -> faz a busca
    param_grid = {'svm__C': 2.**np.arange(-5,15,2), 'svm__gamma': 2.**np.arange(3,-15,-2)}
    grid = GridSearchCV(pipeline, param_grid, n_jobs=-1, verbose=True)
    grid.fit(x_train, y_train)

    # predict -> recupera o melhor modelo
    model = grid.best_estimator_
    y_pred = model.predict(x_test)
    plotResult(y_test, y_pred, labels=model.classes_)
#end main

if __name__ == '__main__':
    main()