#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:41:02 2017

@author: DK
"""

# 1. Data Preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv('/Users/DK/Documents/Machine_Learning/Python-and-R/Machine_Learning_Projects/Artificial_Neural_Networks/Churn_Modelling.csv')
X=dataset.iloc[:,3:13].values
Y=dataset.iloc[:,13].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1=LabelEncoder()
labelencoder_X_2=LabelEncoder()
X[:,1]=labelencoder_X_1.fit_transform(X[:,1])
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])

#Creating dummy variables for column 1
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

#Splitting dataset into test and train
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#feature scaling is mandatory due to the nature of complex calculations that occur in ANNs
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)               #fitting is done on traiing set to get the coefficients and then applieed on the test set


#2. Constructing ANN

#importing libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

#initialise ANN
classifier=Sequential()

#Adding input and first hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))                     #adding nodes to the hidden layer which is usually the avg of the nodes in the input and output layer; In this case the training set is input and has 11 nodes and the ouput has 1 node as it is either 0 or 1 which requires one node

#Adding second hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))

#Adding outer layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#Compiling ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Prediction
classifier.fit(X_train,Y_train,batch_size=10,nb_epoch=100)
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)