# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 01:02:03 2019

@author: praneet jaiin
"""

file_name = "net2.params"
net.save_parameters(file_name)
param=mx.nd.load("net.params")
print(param['0.weight'])
a=param['0.weight']
b=param['0.bias']
a.shape
b.shape
train_images.shape
dense_layer1=np.dot(param['0.weight'],np.transpose(train_images))

from sklearn import LogisticRegressor
log_reg=LogisticRegressor()


param=mx.nd.load("NN2.params")
print(param['0.weight'])
a=param['0.weight']
b=param['0.bias']
a.squeeze()
b.squeeze()
a.shape
b.shape
a.asnumpy()
b.asnumpy()
train_images.shape
a1=param['1.weight']
b1=param['1.bias']
a1.squeeze()
b1.squeeze()
a1.shape
b1.shape
a1.asnumpy()
b1.asnumpy()    
a2=param['2.weight']
b2=param['2.bias']
a2.squeeze()
b2.squeeze()
a2.shape
b2.shape
a2.asnumpy()
b2.asnumpy() 
for i in range(1024):
    print(i)
    for j in range(42000):
        c=a[i,:]
        d=train_images[j,:]
        dense1[i][j]=np.dot(c.T,d)
        print (dense1[i][j])
dense_layer1=np.dot(a,train_images.T)+b.T
dense_layer2=np.dot(a1,dense_layer1)+b1.T
dense_layer3=np.dot(a2,dense_layer2)+b2.T

from sklearn import LogisticRegressor
log_reg=LogisticRegression().fit(np.transpose(dense_layer1),train_labels)
log_reg.score(test_images,test_labels)
