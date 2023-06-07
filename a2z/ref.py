from glob import glob
import sys, os
import cv2
import numpy as np 
import time

from sklearn.preprocessing import minmax_scale
from sklearn.metrics import classification_report

from sklearn import tree

from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def get_pieces(image):
    height, width, _ = image.shape
    
    img1 = image[0:int(height/3) , 0:int(width/3)] # inferior/esquerdo
    img2 = image[int(height/3):int(height/2) , 0:int(width/3)] # esquerdo/meio
    img3 = image[int(height/2):height , 0:int(width/3)] # superior/esquerdo
    
    img4 = image[int(height/2):height , int(width/3):int(width/2)] # cima/meio
    img5 = image[int(height/3):int(height/2) , int(width/3):int(width/2)] # meio
    img6 = image[0:int(height/3) , int(width/3):int(width/2)] # baixo/meio
    
    img7 = image[0:int(height/3) , int(width/2):width] # inferior/direito 
    img8 = image[int(height/3):int(height/2) , int(width/2):width] # direito/meio
    img9 = image[int(height/2):height , int(width/2):width] # superior/direito   
        
    return img1, img2, img3, img4, img5, img6, img7, img8, img9

def rotate_image(image, qtd):
    height, width, _ = image.shape
    point = (width / 2, height / 2)
    
    image_rotacionada = cv2.getRotationMatrix2D(point, qtd, 1.0)
    rotated = cv2.warpAffine(image, image_rotacionada, (width, height))
    
    return get_pieces(rotated)

def main():

    y_test = []
    y_train = []
    X_train = []
    X_test = []
    
    # Conjunto de treinamento
    i = 0
    while i <= 9:
        images = load_images(os.getcwd() + f"/Imagens_Treinamento/{i}")
        for image in images:
            rotules = []            
            for k in range(0,360,45):
                arr = rotate_image(image, k)
                for e in arr:
                    rotules.append(np.sum(e == 0))
                
            y_train.append(i)    
            X_train.append(rotules)
        i += 1
        
    # Conjunto de teste
    i = 0
    while i <= 9:
        images = load_images(os.getcwd() + f"/Imagens_Teste/{i}")
        for image in images:
            rotules = []
            
            for k in range(0,360,45):
                arr = rotate_image(image, k)
                for e in arr:
                    rotules.append(np.sum(e == 0))
                
            y_test.append(i)    
            X_test.append(rotules)
        i += 1
        
    X_train = minmax_scale(X_train)
    X_test = minmax_scale(X_test)
    
    # k-NN classifier
    initial_time = time.time()
    from sklearn.metrics import classification_report
    neigh = KNeighborsClassifier(n_neighbors=13, metric='manhattan')
    neigh.fit(X_train, y_train)
    print('*********************k-NN************************')
    print(classification_report(y_test, neigh.predict(X_test)))
    final_time = time.time()
  
    new_time = final_time - initial_time
    print(f'{new_time} segundos')
  
    # DT - Decision Tree
    initial_time = time.time()
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    print('*********************DT************************')
    print(classification_report(y_test, clf.predict(X_test)))
    final_time = time.time()
  
    new_time = final_time - initial_time
    print(f'{new_time} segundos')
    
    # SVM com Grid search
    initial_time = time.time()
    C_range = 2. ** np.arange(-5,15,2)
    gamma_range = 2. ** np.arange(3,-15,-2)
    k = [ 'rbf']
    # instancia o classificador, gerando probabilidades
    srv = svm.SVC(probability=True, kernel='rbf')
    ss = StandardScaler()
    pipeline = Pipeline([ ('scaler', ss), ('svm', srv) ])
    
    param_grid = {
        'svm__C' : C_range,
        'svm__gamma' : gamma_range
    }
    
    grid = GridSearchCV(pipeline, param_grid, n_jobs=-1, verbose=True)
    grid.fit(X_train, y_train)
    
    model = grid.best_estimator_
    print('*********************SVM************************')
    print(classification_report(y_test, model.predict(X_test)))
    final_time = time.time()
  
    new_time = final_time - initial_time
    print(f'{new_time} segundos')

    # RANDOM FOREST 
    initial_time = time.time()
    clf = RandomForestClassifier(n_estimators=10000, max_depth=30, random_state=1)
    clf.fit(X_train, y_train)  
    print('*********************Random Forest************************')
    print(classification_report(y_test, clf.predict(X_test)))
    final_time = time.time()
  
    new_time = final_time - initial_time
    print(f'{new_time} segundos')
    
    # MLP 
    initial_time = time.time()
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100, 100, 100), random_state=0)
    clf.fit(X_train, y_train)
    print('*********************MLP************************')
    print(classification_report(y_test, clf.predict(X_test)))
    final_time = time.time()
  
    new_time = final_time - initial_time
    print(f'{new_time} segundos')
 
if __name__ == "__main__":
    if len(sys.argv) != 1:
        sys.exit("index.py")

    main()