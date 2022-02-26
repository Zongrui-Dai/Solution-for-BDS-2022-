# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 12:46:47 2022

@author: A
"""


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from keras import optimizers
from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.utils.vis_utils import plot_model

data = np.load('C://Users/A/Documents/Tencent Files/1097968801/FileRecv/mnist.npz')
trainx=data['x_train']
trainy=data['y_train']
testx=data['x_test']
testy=data['y_test']


extract = (trainy==1) | (trainy==7)
trainx = trainx[extract]
trainy = trainy[extract]

extract = (testy==1) | (testy==7)
testx = testx[extract]
testy = testy[extract]


bags_x = np.concatenate((trainx[0:100])).reshape(1,2800,28)
for i in range(100,13000,100):
    a = np.concatenate((trainx[i:i+100])).reshape(1,2800,28)
    bags_x = np.concatenate((bags_x,a),axis=0)

bags_x.resize(130,2800,28,1)

y_train=[]
for i in range(0,13000,100):
    ly = trainy[i:i+100]
    pro = len(ly[ly==1])/100
    
    y_train.append(pro)
    
bags_test = np.concatenate((testx[0:100])).reshape(1,2800,28)
for i in range(100,2100,100):
    a = np.concatenate((testx[i:i+100])).reshape(1,2800,28)
    bags_test = np.concatenate((bags_test,a),axis=0)

bags_test.resize(21,2800,28,1)

y_test=[]
for i in range(0,2100,100):
    ly = testy[i:i+100]
    pro = len(ly[ly==1])/100
    
    y_test.append(pro)

y_train=np.array(y_train)
y_test=np.array(y_test)


############################################################################
adam = optimizers.adam_v2.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999,
                               epsilon=None, decay=0.0005, amsgrad=False)

call = keras.callbacks.EarlyStopping(monitor='val_mae', patience=30, verbose=0, mode='auto', 
                                     baseline=None, restore_best_weights=False)


model = Sequential()
model.add(layers.Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=((2800,28,1))))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(1, (1, 1), activation='relu'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(1, activation='tanh'))

model.compile(loss="mean_absolute_error", optimizer=adam, metrics=['mae'])
hist1 = model.fit(bags_x,y_train, epochs= 100, batch_size=1,callbacks=call, validation_split=0.2)

plt.plot(hist1.history['mae'])
plt.plot(hist1.history['val_mae'])
plt.show()

loss = model.evaluate(bags_test,y_test)
plot_model(model, to_file='./model.png', show_shapes=True)

