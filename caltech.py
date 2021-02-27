from __future__ import absolute_import, division, print_function, unicode_literals

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import numpy as np
import os
from tensorflow.keras import *
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *


#https://github.com/pushkart2/caltech_101_transfer_learning/blob/master/101_object.ipynb7
#https://deeplearning-math.github.io/slides/Project1_WuXuLee.pdf

#https://www.tensorflow.org/tutorials/keras/classification


#https://github.com/charan96/deeplearn-caltech101/blob/master/caltech_convnet.py

"""***Step 1: Train on fashion MNIST***"""

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = np.array(x_train).astype('float')
x_train /= 255.0

x_test = np.array(x_test).astype('float')
x_test /= 255.0

x_train.shape, y_train.shape, x_test.shape, y_test.shape

def get_encoder(input_shape = (28, 28)):
  encoder = Sequential()
  encoder.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
  encoder.add(MaxPooling1D(2))
  encoder.add(Dropout(0.25))

  encoder.add(Conv1D(64, kernel_size=3, activation='relu'))
  encoder.add(MaxPooling1D(pool_size=2))
  encoder.add(Dropout(0.25))

  encoder.add(Conv1D(128, kernel_size=3, activation='relu'))
  encoder.add(Dropout(0.4))
  return encoder

def get_model(encoder, num_layers = 1):
  x = y = Input(shape=(28, 28), name='input_layer')
  y = encoder(y)
  y = Flatten()(y)

  for _ in range(num_layers):
    y = Dense(128, activation='relu')(y)
    y = Dropout(0.3)(y)
  y = Dense(10, activation='softmax')(y)
  return Model(x, y)

# Common encoder for both training
encoder = get_encoder()

# Fashion MNIST model
model = get_model(encoder)
model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
             metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=64, epochs=20, shuffle=True, validation_data=(x_test, y_test))

"""***Step 2: Transfer learning (Fine tuning) - MNIST***"""
# Just train the final layers, rest of the encoder is fixed

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = np.array(x_train).astype('float')
x_train /= 255.0

x_test = np.array(x_test).astype('float')
x_test /= 255.0

# Setting the encoder trainable as false
# This flag is very important because we just want to use the already trained model
encoder.trainable = False

# MNIST model for transfer learning
mnist_model = get_model(encoder, num_layers=2)
mnist_model.summary()
mnist_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
                    metrics=['accuracy'])

mnist_model.fit(x_train, y_train, batch_size=64, epochs=10, shuffle=True, validation_data=(x_test, y_test))

# instructions from: https://michaelblogscode.wordpress.com/2017/10/10/reducing-and-profiling-gpu-memory-usage-in-keras-with-tensorflow-backend/

# TensorFlow wizardry 
# Create a session with the above options specified.
#keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

import os, shutil

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
#os.environ["CUDA_VISIBLE_DEVICES"]="0" 

import numpy as np
import cv2
from matplotlib import pyplot as plt

import keras
print("keras version: ", keras.__version__)

from scipy.io import loadmat
from matplotlib.pyplot import *
import numpy as np
#from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as qda
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import pandas as pd
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#%matplotlib inline




#plt.style.use('ggplot')

from numpy import std, mean, sqrt

import keras
import keras.utils
from keras import utils as np_utils

import tensorflow as tf
print("tensoflow version: ", tf.__version__)

from keras.models import Sequential 
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras import losses, optimizers
from keras.layers import Conv2D, BatchNormalization # we have 2Dimages
from keras.layers import ReLU
import scipy.cluster.hierarchy as sch


from keras import layers
from keras import models 

from keras import optimizers
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from keras.models import load_model

from keras.layers import Dropout
from keras.utils import np_utils
from keras import applications
from keras.preprocessing import image
#from skimage import io, exposure, color, transform
from sklearn.model_selection import train_test_split
import numpy as np
#import cPickle as pickle
import h5py as h5py
import os
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


import numpy as np
import os
#from tensorflow.keras import *
#from tensorflow.keras import backend as K
#from tensorflow.keras.layers import *

from keras.models import load_model


#https://github.com/bhavul/Caltech-101-Object-Classification

image_dataset_path = './101_ObjectCategories'
img_folder_names = []
img_folder_names = [f for f in sorted(os.listdir(image_dataset_path))]
CLASSES =101

print(len(img_folder_names))

model_flag_for_accuracy=0
categories_num = 9
images_number = 9 # images per class shown
selected_image_list = np.random.randint(0, 101, categories_num, dtype='l')

# print categories selected
print('Selected categories:')
print([img_folder_names[i] for i in selected_image_list])


fig, xsubplot = plt.subplots(nrows=9, ncols=9)
fig.set_size_inches(9.5, 8.5)

#plt.subplots_adjust(top=0.85) # to include title on TOP of figure. Otherwise it overlaps due to tight_layout



data_array=[]
data_label=[]
fig.subplots_adjust(wspace=0.1,hspace=0.1)

for i, image_type in enumerate(selected_image_list):
    image_folder_path = image_dataset_path + '/' + img_folder_names[image_type]
    # take the first objects
    image_names = [img for img in sorted(os.listdir(image_folder_path))][:images_number]
    
    for j, image_name in enumerate(image_names):
        image_dir_path = image_folder_path + '/' + image_name
        image = cv2.imread(image_dir_path)
        # resize to 100x100 for all images for this plot
        image = cv2.resize(image, (32, 32)) 
        #aplt.figure()
        data_array.append(image.astype(np.float32))
        folder_path_new =image_folder_path.replace("./101_ObjectCategories/","")
        #print(folder_path_new)
        data_label.append(folder_path_new)
        #print(folder_path_new)
        plt.imshow(image)
        data_array_res1 = np.array(data_array)
        data_array_res2= np.reshape(data_array_res1,(len(data_array_res1),3072))        
        xsubplot[i,j].imshow(image)        
        xsubplot[i,j].set_xticks([])
        xsubplot[i,j].set_yticks([])        
        if j == 0:
            pad = 5 # in points            
            xsubplot[i,j].annotate(img_folder_names[image_type], xy=(0, 0.5), xytext=(-xsubplot[i,j].yaxis.labelpad - pad, 0),
                xycoords=xsubplot[i,j].yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
 
label_encoder_y = LabelEncoder()
data_label_enc = label_encoder_y.fit_transform(data_label)

data_label_enc_cat =  np_utils.to_categorical(data_label_enc,101)

fig.tight_layout()
plt.axis('off')
fig.show()

partitioner = StratifiedShuffleSplit(n_splits=5, train_size=0.8,test_size=0.2,random_state=42)
##############################################################################################

#############################################################################################

def cohen_d_calculation_func(xdata,ydata):
  xlength = len(xdata)
  ylength = len(ydata)
  dof = xlength + ylength - 2
  return (mean(xdata) - mean(ydata)) / sqrt(((xlength-1)*std(xdata, ddof=1) ** 2 + (ylength-1)*std(ydata, ddof=1) ** 2) / dof)

##################    Question 2 #################################################

def Question2():
  #Load the dataset
  diabetes_dataset = pd.read_csv("D:\\downloads\\downloads_19_Nov\\diabetes.csv")
  
  # Lists required for grouping the Dataset.
  firstdataset= []
  seconddataset= []
  #cohen_d = []
  sorted_cohen_d=[]
  #https://cmdlinetips.com/2018/12/how-to-loop-through-pandas-rows-or-how-to-iterate-over-pandas-rows/
  # Iterate through the csv file and extract the required Data.
  for index,item in diabetes_dataset.iterrows():
    if item['Outcome']== 0:
        firstdataset.append(item)        
    else:
        seconddataset.append(item)    
  # Create DataFrame for both the lists.
  print ("Group Not of Interest:")
  print (len(firstdataset))
  print ("Group of Interest:")
  print (len(seconddataset))
  
  dataframewithOutput0 = pd.DataFrame(firstdataset)
  dataframewithOutput1 = pd.DataFrame(seconddataset)
  
  #Loop through the Datatframes, calculate and print the Cohen's d Value.
  #https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/
  for index in range(0,8,1):
    c0=dataframewithOutput0.iloc[:,index]
    c1=dataframewithOutput1.iloc[:,index]
    #cohens_d=(np.mean(c1) - np.mean(c0)) / (np.sqrt((np.std(c0) ** 2 + np.std(c1) ** 2) / 2))    
    cohen_d = cohen_d_calculation_func(c1,c0)
    sorted_cohen_d.append(cohen_d)            
    print("The cohen's d value for:",(index))
    print ("Cohens_d function value= " + str(cohen_d))
  cohen_d = sorted(sorted_cohen_d)
  print(cohen_d)

##############################################################################################
#https://github.com/charan96/deeplearn-caltech101/blob/master/caltech_convnet.py
#Here Transfer learning is applied to improve the model performance.
#The model is trained on other dataset and applied to 101 Caltech Dataset,
#hence basetrained_model.layers is set to False, if it is set to true then performace
#increases.

def transferlearning_cnn_model_architecture(x_train, x_test, y_train, y_test):
  basetrained_model = applications.VGG19(weights='imagenet', include_top=False, input_shape = (32,32,3))  
  
  for layer in basetrained_model.layers:
    layer.trainable = False
	 
  trained_model = Sequential()
  trained_model.add(basetrained_model)
  trained_model.add(Flatten())
  trained_model.add(Dense(256, activation='relu'))
  trained_model.add(Dense(256, activation='relu'))
  trained_model.add(Dense(512, activation='relu'))
  trained_model.add(Dense(512, activation='relu'))
  trained_model.add(Dropout(0.5))
  trained_model.add(Dense(CLASSES, activation='softmax'))
  trained_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
  trained_model.fit(x_train_res, y_train_cat, epochs=20, batch_size=16)
  model_value = trained_model.evaluate(x_test_res,y_test_cat,verbose=0)    
  print ("CNN loss:",model_value[0])
  print ("CNN Accuracy:",model_value[1])  
  error_rate_cnn = 1-model_value[1]
  print ("Error rate CNN:")
  print(error_rate_cnn)
##############################################################################################

def plot_cnn_raw_data():
  basetrained_model = applications.VGG19(weights='imagenet', include_top=False, input_shape = (32,32,3))  
  
  for layer in basetrained_model.layers:
    layer.trainable = False
    
  trained_model = Sequential()
  trained_model.add(basetrained_model)
  trained_model.add(Flatten())
  trained_model.add(Dense(256, activation='relu'))
  trained_model.add(Dense(256, activation='relu'))
  trained_model.add(Dense(512, activation='relu'))
  trained_model.add(Dense(512, activation='relu'))
  trained_model.add(Dropout(0.5))
  trained_model.add(Dense(CLASSES, activation='softmax'))
  trained_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
  plot_data=trained_model.fit(data_array_res1, data_label_enc_cat,validation_split=0.33,epochs=20, batch_size=16)
  model_value = trained_model.evaluate(data_array_res1, data_label_enc_cat,verbose=0)    
  print ("CNN loss:",model_value[0])
  print ("CNN Accuracy:",model_value[1])  
  error_rate_cnn = 1-model_value[1]
  print ("Error rate CNN:")
  print(error_rate_cnn)
  
  acc = plot_data.history['acc']
  val_acc = plot_data.history['val_acc']
  loss = plot_data.history['loss']
  val_loss = plot_data.history['val_loss']

  epochs = range(len(acc))

  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.legend()

  plt.figure()

  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend()
  plt.show()

###############################################################################################
def PCA():
  # Feature Scaling
  sc = StandardScaler()
  X_train = sc.fit_transform(x_train)
  X_test = sc.transform(x_test)

  # Applying PCA
  from sklearn.decomposition import PCA
  pca = PCA(n_components = 63,svd_solver='full')
  X_train = pca.fit_transform(X_train)
  X_test = pca.transform(X_test)
  explained_variance_pca = pca.explained_variance_ratio_
  print(pca.explained_variance_ratio_)
  print(pca.singular_values_)
  
  #PCA Application. 
  # Fitting Logistic Regression to the Training set
  from sklearn.linear_model import LogisticRegression
  classifier = LogisticRegression(random_state = 0)
  classifier.fit(X_train, y_train)
  
  # Predicting the Test set results
  y_pred = classifier.predict(X_test)
  
  
  # Applying PCA
  from sklearn.decomposition import PCA
  pca = PCA(n_components = 62,svd_solver='arpack')
  X_train = pca.fit_transform(X_train)
  X_test = pca.transform(X_test)
  explained_variance_pca = pca.explained_variance_ratio_
  #The explained variance tells us how much information (variance) can be attributed to each of the principal components.
  print(pca.explained_variance_ratio_)
  print(pca.singular_values_)
  
  #PCA Application. 
  # Fitting Logistic Regression to the Training set
  from sklearn.linear_model import LogisticRegression
  classifier = LogisticRegression(random_state = 0)
  classifier.fit(X_train, y_train)
  
  from sklearn.decomposition import KernelPCA
  transformer = KernelPCA(n_components=7, kernel='linear')
  X_transformed_linear = transformer.fit_transform(X_train)
  X_transformed_linear.shape
  
  from sklearn.decomposition import KernelPCA
  transformer = KernelPCA(n_components=7, kernel='poly')
  X_transformed_poly = transformer.fit_transform(X_train)
  X_transformed_poly.shape

  
  transformer = KernelPCA(n_components=7, kernel='rbf')
  X_transformed_rbf = transformer.fit_transform(X_train)
  X_transformed_rbf.shape
  
  transformer = KernelPCA(n_components=7, kernel='sigmoid')
  X_transformed_sigmoid = transformer.fit_transform(X_train)
  X_transformed_sigmoid.shape


###############################################################################################
def svm():
  print(" SVM started") 
  ## Feature Scaling
  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  x_train_ss = sc.fit_transform(x_train)
  x_test_ss = sc.transform(x_test)

  ## Fitting SVM to the Training set
  classifier = SVC(kernel = 'linear', random_state = 0)
  classifier.fit(x_train_ss, y_train)
  
  y_pred = classifier.predict(x_test_ss)
  from sklearn.metrics import classification_report, confusion_matrix
  print(confusion_matrix(y_test,y_pred))
  print(classification_report(y_test,y_pred))

  print("Kernel function poly")
  
  ## Fitting SVM to the Training set
  classifier = SVC(kernel='poly', degree=8)
  classifier.fit(x_train_ss, y_train)
  
  y_pred = classifier.predict(x_test_ss)
  

  print(confusion_matrix(y_test,y_pred))
  print(classification_report(y_test,y_pred))

  print("Kernel function Gaussian")
  
  ## Fitting SVM to the Training set
  classifier = SVC(kernel='rbf')
  classifier.fit(x_train_ss, y_train)
  
  y_pred = classifier.predict(x_test_ss)
  print(confusion_matrix(y_test,y_pred))
  print(classification_report(y_test,y_pred))

  ## Fitting SVM to the Training set  
  classifier = SVC(kernel='sigmoid')
  classifier.fit(x_train_ss, y_train)
  
  y_pred = classifier.predict(x_test_ss)
  
  from sklearn.metrics import classification_report, confusion_matrix
  print(confusion_matrix(y_test,y_pred))
  print(classification_report(y_test,y_pred))
  
################################################################################################
def random_forest_ensemble():
  #print(" RF started")
  # Feature Scaling
  sc = StandardScaler()
  x_train_rf_ss = sc.fit_transform(x_train)
  x_test_rf_ss = sc.transform(x_test)

  # Fitting Random Forest Classification to the Training set  
  classifier = RandomForestClassifier(n_estimators = 64, criterion = 'entropy', random_state = 0)
  classifier.fit(x_train_rf_ss, y_train)

  # Predicting the Test set results
  y_pred_rf = classifier.predict(x_test_rf_ss)
  
  classifier = DecisionTreeClassifier(random_state=0)
  classifier.fit(x_train_rf_ss, y_train)
  
   # Predicting the Test set results
  y_pred_rf = classifier.predict(x_test_rf_ss)
  
  #print ("RF completed")
########################################################################################################

def K_means_clustering():
  print(" Kmeans Started")
  # Using the elbow method to find the optimal number of clusters
  wcss = []
  for i in range(1, 64):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(data_array_res2)
    wcss.append(kmeans.inertia_)
  plt.plot(wcss)
  plt.title('The Elbow Method')
  plt.xlabel('Number of clusters')
  plt.ylabel('WCSS')
  plt.show()

  # Fitting K-Means to the dataset
  kmeans = KMeans(n_clusters = 63, init = 'k-means++', random_state = 42)
  y_kmeans = kmeans.fit_predict(data_array_res2)
#######################################################################################################################
def hierarchial_clustering():
  #print(" Dendogram Cluster Hierarchy started")
  # Using the dendrogram to find the optimal number of clusters
  
  dendrogram = sch.dendrogram(sch.linkage(x_train, method = 'ward'))

  #print(" Dendogram Cluster Hierarchy completed")

  #print(" Dendogram Cluster Hierarchy Plot started")
  # Fitting Hierarchical Clustering to the dataset
  from sklearn.cluster import AgglomerativeClustering
  hc = AgglomerativeClustering(n_clusters = 64, affinity = 'euclidean', linkage = 'ward')
  y_hc = hc.fit_predict(x_train)

  # Visualising the clusters
  plt.scatter(x_train[y_hc == 0, 0], x_train[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
  plt.scatter(x_train[y_hc == 1, 0], x_train[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
  plt.scatter(x_train[y_hc == 2, 0], x_train[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
  plt.scatter(x_train[y_hc == 3, 0], x_train[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
  plt.scatter(x_train[y_hc == 4, 0], x_train[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
  plt.scatter(x_train[y_hc == 5, 0], x_train[y_hc == 5, 1], s = 100, c = 'red', label = 'Cluster 6')
  plt.scatter(x_train[y_hc == 6, 0], x_train[y_hc == 6, 1], s = 100, c = 'blue', label = 'Cluster 7')
  plt.scatter(x_train[y_hc == 7, 0], x_train[y_hc == 7, 1], s = 100, c = 'green', label = 'Cluster 8')
  plt.scatter(x_train[y_hc == 8, 0], x_train[y_hc == 8,1], s = 100, c = 'cyan', label = 'Cluster 9')
  plt.scatter(x_train[y_hc == 9, 0], x_train[y_hc == 9, 1], s = 100, c = 'magenta', label = 'Cluster 10')
  plt.scatter(x_train[y_hc == 10, 0], x_train[y_hc == 10, 1], s = 100, c = 'red', label = 'Cluster 11')
  plt.scatter(x_train[y_hc == 11, 0], x_train[y_hc == 11, 1], s = 100, c = 'blue', label = 'Cluster 12')
  plt.scatter(x_train[y_hc == 12, 0], x_train[y_hc == 12, 1], s = 100, c = 'green', label = 'Cluster 13')
  plt.scatter(x_train[y_hc == 13, 0], x_train[y_hc == 13, 1], s = 100, c = 'cyan', label = 'Cluster 14')
  plt.scatter(x_train[y_hc == 14, 0], x_train[y_hc == 14, 1], s = 100, c = 'magenta', label = 'Cluster 15')
  plt.scatter(x_train[y_hc == 15, 0], x_train[y_hc == 15, 1], s = 100, c = 'red', label = 'Cluster 16')
  plt.scatter(x_train[y_hc == 16, 0], x_train[y_hc == 16, 1], s = 100, c = 'blue', label = 'Cluster 17')
  plt.scatter(x_train[y_hc == 17, 0], x_train[y_hc == 17, 1], s = 100, c = 'green', label = 'Cluster 18')
  plt.scatter(x_train[y_hc == 18, 0], x_train[y_hc == 18, 1], s = 100, c = 'cyan', label = 'Cluster 19')
  plt.scatter(x_train[y_hc == 19, 0], x_train[y_hc == 19, 1], s = 100, c = 'magenta', label = 'Cluster 20')
  plt.scatter(x_train[y_hc == 20, 0], x_train[y_hc == 20, 1], s = 100, c = 'red', label = 'Cluster 21')
  plt.scatter(x_train[y_hc == 21, 0], x_train[y_hc == 21, 1], s = 100, c = 'blue', label = 'Cluster 22')
  plt.scatter(x_train[y_hc == 22, 0], x_train[y_hc == 22, 1], s = 100, c = 'green', label = 'Cluster 23')
  plt.scatter(x_train[y_hc == 23, 0], x_train[y_hc == 23, 1], s = 100, c = 'cyan', label = 'Cluster 24')
  plt.scatter(x_train[y_hc == 24, 0], x_train[y_hc == 24, 1], s = 100, c = 'magenta', label = 'Cluster 25')
  plt.scatter(x_train[y_hc == 25, 0], x_train[y_hc == 25, 1], s = 100, c = 'red', label = 'Cluster 26')
  plt.scatter(x_train[y_hc == 26, 0], x_train[y_hc == 26, 1], s = 100, c = 'blue', label = 'Cluster 27')
  plt.scatter(x_train[y_hc == 27, 0], x_train[y_hc == 27, 1], s = 100, c = 'green', label = 'Cluster 28')
  plt.scatter(x_train[y_hc == 28, 0], x_train[y_hc == 28, 1], s = 100, c = 'cyan', label = 'Cluster 29')
  plt.scatter(x_train[y_hc == 29, 0], x_train[y_hc == 29, 1], s = 100, c = 'magenta', label = 'Cluster 30')
  plt.scatter(x_train[y_hc == 30, 0], x_train[y_hc == 30, 1], s = 100, c = 'red', label = 'Cluster 31')
  plt.scatter(x_train[y_hc == 31, 0], x_train[y_hc == 31, 1], s = 100, c = 'blue', label = 'Cluster 32')
  plt.scatter(x_train[y_hc == 32, 0], x_train[y_hc == 32, 1], s = 100, c = 'green', label = 'Cluster 33')
  plt.scatter(x_train[y_hc == 33, 0], x_train[y_hc == 33, 1], s = 100, c = 'cyan', label = 'Cluster 34')
  plt.scatter(x_train[y_hc == 34, 0], x_train[y_hc == 34, 1], s = 100, c = 'magenta', label = 'Cluster 35')
  plt.scatter(x_train[y_hc == 35, 0], x_train[y_hc == 35, 1], s = 100, c = 'red', label = 'Cluster 16')
  plt.scatter(x_train[y_hc == 36, 0], x_train[y_hc == 36, 1], s = 100, c = 'blue', label = 'Cluster 17')
  plt.scatter(x_train[y_hc == 37, 0], x_train[y_hc == 37, 1], s = 100, c = 'green', label = 'Cluster 18')
  plt.scatter(x_train[y_hc == 38, 0], x_train[y_hc == 38, 1], s = 100, c = 'cyan', label = 'Cluster 19')
  plt.scatter(x_train[y_hc == 39, 0], x_train[y_hc == 39, 1], s = 100, c = 'magenta', label = 'Cluster 20')
  plt.scatter(x_train[y_hc == 40, 0], x_train[y_hc == 40, 1], s = 100, c = 'red', label = 'Cluster 21')
  plt.scatter(x_train[y_hc == 41, 0], x_train[y_hc == 41, 1], s = 100, c = 'blue', label = 'Cluster 22')
  plt.scatter(x_train[y_hc == 42, 0], x_train[y_hc == 42, 1], s = 100, c = 'green', label = 'Cluster 23')
  plt.scatter(x_train[y_hc == 43, 0], x_train[y_hc == 43, 1], s = 100, c = 'cyan', label = 'Cluster 24')
  plt.scatter(x_train[y_hc == 44, 0], x_train[y_hc == 44, 1], s = 100, c = 'magenta', label = 'Cluster 25')
  plt.scatter(x_train[y_hc == 45, 0], x_train[y_hc == 45, 1], s = 100, c = 'red', label = 'Cluster 26')
  plt.scatter(x_train[y_hc == 46, 0], x_train[y_hc == 46, 1], s = 100, c = 'blue', label = 'Cluster 27')
  plt.scatter(x_train[y_hc == 47, 0], x_train[y_hc == 47, 1], s = 100, c = 'green', label = 'Cluster 28')
  plt.scatter(x_train[y_hc == 48, 0], x_train[y_hc == 48, 1], s = 100, c = 'cyan', label = 'Cluster 29')
  plt.scatter(x_train[y_hc == 49, 0], x_train[y_hc == 49, 1], s = 100, c = 'magenta', label = 'Cluster 30')
  plt.scatter(x_train[y_hc == 50, 0], x_train[y_hc == 50, 1], s = 100, c = 'red', label = 'Cluster 31')
  plt.scatter(x_train[y_hc == 51, 0], x_train[y_hc == 51, 1], s = 100, c = 'blue', label = 'Cluster 32')
  plt.scatter(x_train[y_hc == 52, 0], x_train[y_hc == 52, 1], s = 100, c = 'green', label = 'Cluster 33')
  plt.scatter(x_train[y_hc == 53, 0], x_train[y_hc == 53, 1], s = 100, c = 'cyan', label = 'Cluster 34')
  plt.scatter(x_train[y_hc == 54, 0], x_train[y_hc == 54, 1], s = 100, c = 'magenta', label = 'Cluster 35')
  plt.scatter(x_train[y_hc == 55, 0], x_train[y_hc == 55, 1], s = 100, c = 'red', label = 'Cluster 36')
  plt.scatter(x_train[y_hc == 56, 0], x_train[y_hc == 56, 1], s = 100, c = 'blue', label = 'Cluster 37')
  plt.scatter(x_train[y_hc == 57, 0], x_train[y_hc == 57, 1], s = 100, c = 'green', label = 'Cluster 38')
  plt.scatter(x_train[y_hc == 58, 0], x_train[y_hc == 58, 1], s = 100, c = 'cyan', label = 'Cluster 39')
  plt.scatter(x_train[y_hc == 59, 0], x_train[y_hc == 59, 1], s = 100, c = 'magenta', label = 'Cluster 40')
  plt.scatter(x_train[y_hc == 60, 0], x_train[y_hc == 60, 1], s = 100, c = 'red', label = 'Cluster 41')
  plt.scatter(x_train[y_hc == 61, 0], x_train[y_hc == 61, 1], s = 100, c = 'blue', label = 'Cluster 42')
  plt.scatter(x_train[y_hc == 62, 0], x_train[y_hc == 62, 1], s = 100, c = 'green', label = 'Cluster 43')
  plt.scatter(x_train[y_hc == 63, 0], x_train[y_hc == 63, 1], s = 100, c = 'cyan', label = 'Cluster 44')
  plt.scatter(x_train[y_hc == 64, 0], x_train[y_hc == 64, 1], s = 100, c = 'magenta', label = 'Cluster 45')
    
  plt.title('Clusters')
  plt.xlabel('Train data')
  plt.ylabel('Train label')
  plt.legend()
  plt.show()

  #print(" Dendogram Cluster Hierarchy Plot completed")

########################################################################################################################
#
#
########################################################################################################

############################   ANN with 3 Dense layers ###############################################  
def ann():
  # Initialising the ANN
  classifier = Sequential()

  # Adding the fourth hidden layer  
  classifier.add(Dense(units = 512, kernel_initializer = 'uniform', activation = 'relu', input_dim = 3072))
  
  # Adding the fifth hidden layer
  classifier.add(Dense(units = 512, kernel_initializer = 'uniform', activation = 'relu', input_dim = 3072))
  
  # Adding the output layer  
  classifier.add(Dense(units = 101, kernel_initializer = 'uniform', activation = 'softmax'))
  
  # Compiling the ANN
  # With Optimizer as rmsprop, model performed well compared to optimizer as adam. 
  #classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
  classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
  
  # Fitting the ANN to the Training set
  trainingmodel2=classifier.fit(x_train,y_train_cat,epochs=50,shuffle = True,validation_split=0.2)
      
  model_value = classifier.evaluate(x_test,y_test_cat,verbose=0)

  print ("ANN loss:",model_value[0])
  print ("ANN Accuracy:",model_value[1])
  
  error_rate_ann = 1-model_value[1]
  print ("Error rate ANN:")
  print(error_rate_ann)
  
#########################################################################################################

def cnn_model_save_method(model_flag_for_accuracy):
  print(" Model building started ")  
  if(model_flag_for_accuracy==0):     
    model = Sequential()  
    # BatchNormalization() normalizes the Activation of the previous layer but 
    #invoking after each Convolution or Max Pooling layer, Accuracy remained 
    #constant and Accuracy didn't increase hence removed them.
    model.add(BatchNormalization())   
    # Step 1 - Convolution
    model.add(Convolution2D(32,(3, 3), input_shape = (28, 28, 1), activation = 'relu',data_format="channels_last"))  
    # Filters specified as 32 provided less Accuracy compared to Filters provided as 64 
    #and this helped reduce couple of hidden layers with 32 value kernel filter.    
    model.add(Convolution2D(64, (3, 3), activation = 'relu',padding='same'))   
  
    model.add(MaxPooling2D(pool_size = (2, 2))) 
  
    # Adding a third convolutional layer  
    model.add(Convolution2D(32,(3, 3), activation = 'relu'))

    # Adding a third convolutional layer  
    model.add(Convolution2D(64,(3, 3), activation = 'relu'))

     #Adding a third convolutional layer  
    model.add(Convolution2D(128,(3, 3), activation = 'relu'))
              
    # Step 3 - Flattening
    model.add(Flatten())
    
    # Step 4 - Full connection
    model.add(Dense(activation = 'relu',units = 512 ))
    model.add(Dense(activation = 'relu',units = 320 ))
    model.add(Dense(activation = 'softmax',units = 101 ))  
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    plot_data= model.fit(x_train_res,y_train_cat,batch_size=32,validation_data=(x_test_res,y_test_cat),epochs=50)
    model_value = model.evaluate(x_test_res,y_test_cat,verbose=0)

    print ("CNN loss:",model_value[0])
    print ("CNN Accuracy:",model_value[1])  
    error_rate_cnn = 1-model_value[1]
    print ("Error rate CNN:")
    print(error_rate_cnn)
  else :
    model = load_model('caltech101_1.h5')        
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.fit(x_train_res,y_train_cat,batch_size=32,validation_data=(x_test_res,y_test_cat),epochs=50)
    model_value = model.evaluate(x_test_res,y_test_cat,verbose=0)
    print("Inside load_model")        
        
  model.summary()
  model.save('caltech101_1.h5')
  print("Completed cnn")
#################################################################################################

# This code has been implemented by Derek and checked into Moodle for Assignment 
# reference purpose.And the same code has been used and modified here to calculate Error rate and Accuracy for QDA.  
def StratifiedShuffleSplit_cross_validate_func_QDA(X, y,partitioner) -> (np.array, np.array,np.array):    
    runs = 4
    accuracy_list=[]
    error_rate_list=[]
    QDA= np.empty([runs])
    for i in range(runs):        
        qda_results = cross_validate(qda(), X, y, scoring="accuracy", cv=partitioner)        
        QDA[i] = np.mean(qda_results["test_score"])
        error_rate_qda = 1-QDA[i] 
        print("QDA[i]")
        print(QDA[i])        
        print("error_rate_qda")
        print(error_rate_qda)
        accuracy_list.append(QDA[i])
        error_rate_list.append(error_rate_qda)
    plt.plot(error_rate_list)
    plt.show()
    plt.plot(accuracy_list)
    plt.show()

########################################################################################################################      
# This code has been implemented by Derek and checked into Moodle for Assignment 
# reference purpose.And the same code has been used and modified here to calculate Error rate and Accuracy for lda.  
def StratifiedShuffleSplit_cross_validate_func_lda(X, y,partitioner) -> (np.array, np.array,np.array):    
    runs = 4
    lDA= np.empty([runs])
    accuracy_list=[]
    error_rate_list=[]
    for i in range(runs):        
        lda_results = cross_validate(lda(), X, y, scoring="accuracy", cv=partitioner)        
        lDA[i] = np.mean(lda_results["test_score"])
        error_rate_lda = 1-lDA[i]
        print("lDA[i]")
        print(lDA[i])        
        print("error_rate_lda")
        print(error_rate_lda)
        accuracy_list.append(lDA[i])
        error_rate_list.append(error_rate_lda)
    plt.plot(error_rate_list)
    plt.show()
    plt.plot(accuracy_list)
    plt.show()          


##########################################################################################################################
# This code has been implemented by Derek and checked into Moodle for Assignment 
# reference purpose.And the same code has been used and modified here to calculate Error rate and Accuracy for NaiveBayes.  
def StratifiedShuffleSplit_cross_validate_func_NaiveBayes(X, y,partitioner) -> (np.array, np.array,np.array):    
    runs = 4
    accuracy_list=[]
    error_rate_list=[]
    NaiveBayes= np.empty([runs])
    for i in range(runs):        
        NaiveBayes_results = cross_validate(NB(), X, y, scoring="accuracy", cv=partitioner)
        
        NaiveBayes[i] = np.mean(NaiveBayes_results["test_score"])
        error_rate_nb = 1-NaiveBayes[i]
        print("NaiveBayes[i]")
        print(NaiveBayes[i])                        
        print("error_rate_nb")
        print(error_rate_nb)
        accuracy_list.append(NaiveBayes[i])
        error_rate_list.append(error_rate_nb)
    plt.plot(error_rate_list)
    plt.show()
    plt.plot(accuracy_list)
    plt.show()         

##################################################################################################################

# This code has been implemented by Derek and checked into Moodle for Assignment 
# reference purpose.And the same code has been used and modified here to calculate Error rate and Accuracy for KNN.  
def StratifiedShuffleSplit_cross_validate_func_knn(X, y,partitioner) -> (np.array, np.array,np.array):
    
    runs = 4
    accuracy_list=[]
    error_rate_list=[]
    Knn,Knn1,Knn2= np.empty([runs]),np.empty([runs]),np.empty([runs])
    for i in range(runs):        
        Knn_results = cross_validate(KNeighborsClassifier(n_neighbors=1, n_jobs=-1), X, y, scoring="accuracy", cv=partitioner)
        Knn_results1 = cross_validate(KNeighborsClassifier(n_neighbors=5, n_jobs=-1), X, y, scoring="accuracy", cv=partitioner)
        Knn_results2 = cross_validate(KNeighborsClassifier(n_neighbors=10, n_jobs=-1), X, y, scoring="accuracy", cv=partitioner)
        
        Knn[i] =  np.mean(Knn_results["test_score"])
        Knn1[i] = np.mean(Knn_results1["test_score"])
        Knn2[i] = np.mean(Knn_results2["test_score"])
        accuracy_list.append(Knn[i])
        accuracy_list.append(Knn1[i])
        accuracy_list.append(Knn2[i])
        #Calculating Error rate.
        error_rate_knn1 = 1-Knn[i]
        error_rate_knn5 = 1-Knn1[i]
        error_rate_knn10 = 1-Knn2[i]
        error_rate_list.append(error_rate_knn1)
        error_rate_list.append(error_rate_knn5)
        error_rate_list.append(error_rate_knn10)
        
        print("Knn[i]")
        print(Knn[i])
        print("Knn1[i]")
        print(Knn1[i])
        print("Knn2[i]")
        print(Knn2[i])
        print("error_rate_knn1")
        print(error_rate_knn1)        
        print("error_rate_knn5")
        print(error_rate_knn5)
        print("error_rate_knn10")
        print(error_rate_knn10)
    plt.plot(error_rate_list)
    plt.show()
    plt.plot(accuracy_list)
    plt.show()  
#########################################################################################

#print(np.unique(data_label,return_inverse=True)[1])
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
sss.get_n_splits(data_array_res2, data_label_enc)


for train_index, test_index in sss.split(data_array_res2, data_label_enc):
  print("TRAIN:", train_index, "TEST:", test_index)
  x_train, x_test = data_array_res2[train_index], data_array_res2[test_index]
  y_train, y_test = data_label_enc[train_index], data_label_enc[test_index]    
  
  x_train_res = x_train.reshape(len(x_train),32,32,3)
  x_test_res = x_test.reshape(len(x_test),32,32,3)

  y_train_res = y_train.reshape((len(y_train),))
  y_test_res= y_test.reshape((len(y_test),))

  y_train_cat =  np_utils.to_categorical(y_train,101)
  y_test_cat =   np_utils.to_categorical(y_test,101)
  Question2()
  transferlearning_cnn_model_architecture(x_train,y_train,x_test,y_test)
  cnn_model_save_method(model_flag_for_accuracy)
  plot_cnn_raw_data()
  model_flag_for_accuracy=1
  StratifiedShuffleSplit_cross_validate_func_QDA(x_train,y_train,partitioner)
  StratifiedShuffleSplit_cross_validate_func_lda(x_train,y_train,partitioner)
  StratifiedShuffleSplit_cross_validate_func_NaiveBayes(x_train,y_train,partitioner)
  StratifiedShuffleSplit_cross_validate_func_knn(x_train,y_train,partitioner)
  svm()
  random_forest_ensemble()
  K_means_clustering()
  hierarchial_clustering()  
  PCA()
  ann() 
  
##########################################################################################

