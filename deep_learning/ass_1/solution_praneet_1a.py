# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 09:00:37 2019

@author: praneet jain
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df= pd.read_csv('Concrete_Data.csv')
X = df.iloc[:,:-1].values
Y = df.iloc[:,-1].values

rows,cols=X.shape[0],X.shape[1]
print (rows,cols)

np.random.seed(48)

def shuffle_dataset(X,Y):
    np.random.shuffle(X,Y)
    pass

training_size = int(0.8*rows)
X_train = X[:training_size]
y_train = Y[:training_size]
X_test = X[training_size:]
y_test = Y[training_size:]



class LinearRegression(object):
    def __init__(self):
        #Initialize all parameters
        
        self.w = np.random.randn(cols)#? Sample an array corresponding to the number of input features (cols) from a uniform distribution between -1 and 1
        self.b = np.random.randn(1)
        print (self.w.shape)#? Sample from a uniform distribution between -1 and 1
    
    def forward(self, x):
        y=np.multiply(x,self.w)
        y=y+self.b
        y=y.sum()
        return y
        raise NotImplementedError
        
    
    def backward(self, x, ypred, y_train, lr):
        a=ypred-y_train
        for i in range(8):
            self.w[i]=self.w[i]-lr*a*x[i]
 
def MSELoss(y, ypred):
    MSE=(y-ypred)*(y-ypred)
    MSE.shape
    MSE=MSE/2
    return MSE
    print (MSE)
    raise NotImplementedError       
print('Starting Training with Gradient Descent')
lreg = LinearRegression()
epochs = 100
learning_rate = 0.0000001

loss_history = []
epoch_history = []

for e in range(epochs):
    print("epoch no.",e)
    for i in range(824):
        ypred = lreg.forward(X_train[i]) 
        loss = MSELoss(y_train[i], ypred)
        #if(i%3==0):
           # print(loss , ypred , y_train[i])
        if e==0 or (e+1)%100==0:
            loss_history.append(loss)
            epoch_history.append(e+1)
            
        lreg.backward(X_train[i], ypred, y_train[i], learning_rate)
print(loss)
print('Loss fuction decrease after ' + str(epochs) + ' epochs of training')
plt.plot(epoch_history, loss_history)
plt.show()

print('Final training loss')   
y_train_loss= 1# Print training loss ?
print('Starting to test')
ytest_pred=1 # find predictions on test set ?
loss=1 # compute loss on test set ?
print('Final test loss: ' + str(loss))


