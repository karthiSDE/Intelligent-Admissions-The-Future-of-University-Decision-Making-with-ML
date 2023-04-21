import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from scipy import stats
# %matplotlib inline
data = pd.read_csv('Flask\Admission_Predict.csv')
# data.info()
# data.isnull().any()
data=data.rename(columns = {'Chance of Admit ':'Chance of Admit'})
data=data.rename(columns = {'LOR ':'LOR'})
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
x=data.iloc[:,0:7].values
y=data.iloc[:,7:].values
# print(y)
x=sc.fit_transform(x)
# print(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30,random_state=101)
y_train=(y_train>0.5)
# print(y_train)
y_test=(y_test>0.5)
# print(y_test)
from sklearn.linear_model import LogisticRegression
cls =LogisticRegression(random_state =0)
lr=cls.fit(x_train, y_train.argmax(axis=1))
y_pred =lr.predict(x_test)
# print(y_pred)
#libraries to train neural networks
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
#initialize the model
model=keras.Sequential()
#Add input layer
model.add(Dense(7,activation ='relu',input_dim=7))
#Add hidden layer
model.add(Dense(7,activation='relu'))
#Add output layer
model.add(Dense(1,activation='linear'))
model.summary()
model: "sequential"
model.compile(loss ='binary_crossentropy', optimizer = 'adam',
metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 20, epochs = 100)
from sklearn.metrics import accuracy_score
#make predictions on the training data
train_predictions = model.predict(x_train)
# print(train_predictions)
#get the training accuracy
train_acc = model.evaluate(x_train, y_train, verbose=0)[1]
# print(train_acc)
#get the test accuracy
test_acc = model.evaluate(x_test, y_test, verbose=0)[1]
# print(test_acc)
pred=model.predict(x_test)
pred = (pred>0.5)
# print(pred)
y_pred = y_pred.astype(int)
# print(y_pred)
y_test = y_test.astype(int)
# print(y_test)
model.save('model.h5')
