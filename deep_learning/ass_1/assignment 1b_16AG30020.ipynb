# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:50:22 2019

@author: praneet jaiin
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df= pd.read_csv('Iris_Data.csv')



X = df.iloc[:,:-1].values
Y = df.iloc[:,-1].values


rows,cols=  X.shape[0],X.shape[1]

print(rows,cols)

np.random.seed(42) 

def shuffle_dataset(X,Y):
    
    s=np.array(range(X.shape[0]))
    np.random.shuffle(s)
    Y_new=np.zeros((rows,3))
    Y_new[:,0]=(Y==0)
    Y_new[:,1]=(Y==1)
    Y_new[:,2]=(Y==2)
    return X[s].T,Y_new[s].T
    pass
X,Y=shuffle_dataset(X,Y)
training_size = int(0.8*rows)
X_train = X[:,:training_size]
y_train = Y[:,:training_size]
X_test = X[:,training_size:]
y_test = Y[:,training_size:]

class NeuralNetwork(object):
    
    def __init__(self, input_no, hidden_no, output_no ):
        self.h=   np.zeros((10,1))
        self.w1=  np.random.randn(10,4)*0.01
        self.b1=  np.random.randn(10,1)*0.1
        self.w2=  np.random.randn(3,10)*0.01
        self.b2=  np.random.randn(3,1)*0.01
        pass
    
    def relu(self,x):
        np.maximum(x, 0)
        return x
    
    def softmax(self,x):
         expA = np.exp(x)
         return expA / expA.sum()
        
    def forward(self, x):
        y_pred=[]
        self.h=np.maximum((np.dot(self.w1,x)+self.b1),0)
        s=np.dot(self.w2,self.h)+self.b2
        s=s-np.amax(s,axis=0,keepdims=True)
        e_x=np.exp(s)
        y_pred=e_x/np.sum(e_x,axis=0,keepdims=True)
        return y_pred
        pass
    
    def backward(self, x, y_train, y_pred, lr):
        diff_z_2=y_pred-y_train
        dW2=np.dot(diff_z_2,self.h.T)/x.shape[1]
        db2=np.sum(diff_z_2,axis=1,keepdims=True)/x.shape[1]
        Z1=np.dot(self.w1,x)+self.b1
        diff_z_1=np.dot(self.w2.T,diff_z_2)*(Z1>=0)
        dW1=np.dot(diff_z_1,x.T)/x.shape[1]
        db1=np.sum(diff_z_1,axis=1,keepdims=True)/x.shape[1]
        self.w1=self.w1-lr*dW1
        self.w2=self.w2-lr*dW2
        self.b1=self.b1-lr*db1
        self.b2=self.b2-lr*db2
        pass
                    
def crossEntropy_loss(y_pred, y_train):
    loss = 0
    loss= -np.sum(y_train*np.log(y_pred))/y_train.shape[1]
    return loss
    pass

def accuracy(y_pred,y_train):
    acc=0
    y_1=np.argmax(y_pred,axis=1)
    y_2=np.argmax(y_train,axis=1)
    acc=np.sum(y_1==y_2)/y_pred.shape[1]
    return acc
    pass
        
nnobj= NeuralNetwork(cols,10,3)       
epochs = 10000
learning_rate = 1e-2
loss_history = []
epoch_history = []

for e in range(epochs):
    yPred= nnobj.forward(X_train)
    nnobj.backward(X_train, y_train,yPred, lr=learning_rate)
    loss=crossEntropy_loss(yPred, y_train)
    if e==0 or (e+1)%100==0:
        loss_history.append(loss)
        epoch_history.append(e+1)
        
train_loss= crossEntropy_loss(nnobj.forward(X_train), y_train)
train_accuracy= accuracy(nnobj.forward(X_train), y_train)
test_loss= crossEntropy_loss(nnobj.forward(X_test), y_test)
test_accuracy= accuracy(nnobj.forward(X_test), y_test)
    
print("Final train_loss "+ str(train_loss))    
print("Final train_accuracy "+ str(train_accuracy))    
print("Testloss " + str(test_loss))
print("Accuracy is "+ str(test_accuracy))
