import numpy as np
import os
from scipy.misc import imread

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = imread(os.path.join(folder,filename)).reshape(3,100,100)
        #print img.shape, filename
        if img is not None:
            images.append(img)
    return images

def get_pancake_data():
    # load and label pancakes
    pancakes = np.array(load_images_from_folder('data/pancake-pics'))
    pancake_label = np.ones(len(pancakes))
    print 'Postive example shape: ', pancakes.shape
    
    # load and label non pancakes
    not_pancakes = np.array(load_images_from_folder('data/not-pancake-pics'))
    print 'Negative example shape: ', not_pancakes.shape
    not_pancake_label = np.zeros(len(not_pancakes))
   
    X = np.concatenate((pancakes,not_pancakes),axis=0)
    y = np.concatenate((pancake_label,not_pancake_label),axis=0)
    
    randindices     = np.random.choice(X.shape[0],X.shape[0]*0.8,replace=False)
    
    data = {}
    
    mask = np.zeros(X.shape[0], dtype=bool)
    mask[randindices] = True
    
    data['X_train'] = X[mask]
    data['y_train'] = y[mask]    
    
    mask = np.ones(X.shape[0], dtype=bool)
    mask[randindices] = False
    data['X_val']   = X[mask]
    data['y_val']   = y[mask]
    
    print 'X_train shape: ', data['X_train'].shape
    print 'y_train shape: ', data['y_train'].shape
    print 'X_val shape: ', data['X_val'].shape
    print 'y_val shape: ', data['y_val'].shape

    return data