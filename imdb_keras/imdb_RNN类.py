# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 21:25:51 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 22:13:37 2018

@author: 1
"""

import keras
import numpy as np
from keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data()

####每个评论多少个字
ave_len=list(map(len,X_train))
print(np.mean(ave_len))

import matplotlib.pyplot as plt
plt.hist(ave_len,bins=range(min(ave_len),max(ave_len)+50,50))
plt.show()


from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

########截取字符
maxword=400
X_train=sequence.pad_sequences(X_train,maxlen=maxword)
X_test=sequence.pad_sequences(X_test,maxlen=maxword)
vocab_size=np.max([np.max(X_train[i]) for i in range(X_train.shape[0])])+1

                 

def gru_model():
    model = Sequential()
    model.add(GRU(50, input_shape = (300,1), return_sequences = True))
    model.add(GRU(1, return_sequences = False))
    model.add(Activation('sigmoid'))
    
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model
model = gru_model()
model.fit(X_train, y_train, batch_size = 100, epochs = 10, verbose = 0)                 
#########搭建GRU模型
from keras.layers.recurrent import GRU
from keras.layers.core import Dense,Activation,Dropout


model=Sequential()
model.add(GRU(50, input_shape = (300,1), return_sequences = True))
model.add(GRU(1, return_sequences = False))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, batch_size = 100, epochs = 10, verbose = 0)

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=5,batch_size=100)
scores=model.evaluate(X_test,y_test)
print(scores)



model.add(Embedding(vocab_size,64,input_length=maxword))
model.add(GRU(128,return_sequences=True))
model.add(Dropout(0.15))
model.add(GRU(64,return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(32,return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=5,batch_size=100)
scores=model.evaluate(X_test,y_test)
print(scores)

##########搭建LSTM
from keras.layers import LSTM
model=Sequential()
model.add(Embedding(vocab_size,64,input_length=maxword))
model.add(LSTM(128,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=5,batch_size=100)
scores=model.evaluate(X_test,y_test)
print(scores)

####搭建MLP模型
model=Sequential()
model.add(Embedding(vocab_size,64,input_length=maxword))
model.add(Flatten())
model.add(Dense(2000,activation='relu'))
model.add(Dense(500,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=20,batch_size=100,verbose=1)
score=model.evaluate(X_test,y_test)


##########搭建CNN
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv1D,MaxPooling1D
model=Sequential()
model.add(Embedding(vocab_size,64,input_length=maxword))
model.add(Conv1D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv1D(filters=128,kernel_size=3,padding='same',activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=20,batch_size=100)
score=model.evaluate(X_test,y_test)







































