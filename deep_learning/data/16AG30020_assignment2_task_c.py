# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 15:57:06 2019

@author: praneet jaiin
"""

from zipfile import ZipFile
import numpy as np
import mxnet as mx
from mxnet import nd , autograd , gluon
import numpy as np

class DataLoader(object):
    def __init__(self):
        DIR = '../data/'
        pass
        
    def load_data(self, mode = 'train'):
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = '../data/' + label_filename + '.zip'
        image_zip = '../data/' + image_filename + '.zip'
        with ZipFile(label_zip, 'r') as lblzip:
            labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
        with ZipFile(image_zip, 'r') as imgzip:
            images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        return images, labels
    def load_data_test(self, mode = 'test'):
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = '../data/' + label_filename + '.zip'
        image_zip = '../data/' + image_filename + '.zip'
        with ZipFile(label_zip, 'r') as lblzip:
            labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
        with ZipFile(image_zip, 'r') as imgzip:
            images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        return images, labels

    def create_batches(self):
        pass
    
Dataset=DataLoader()
train_images,train_labels=Dataset.load_data()
test_images,test_labels=Dataset.load_data_test()
Batch_size=128


from multiprocessing import cpu_count
CPU_COUNT = cpu_count()

ctx = mx.gpu() 
data_ctx = ctx
model_ctx = ctx
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(1024, activation="relu"))
    net.add(gluon.nn.Dense(512, activation="relu"))
    net.add(gluon.nn.Dense(256, activation="relu"))
    net.add(gluon.nn.Dense(10))
    
    
net.collect_params().initialize(mx.init.Normal(sigma=.01), ctx=model_ctx)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)


def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


epochs = 50
smoothing_constant = .001
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .001})
training_loss_vector=[]
validation_loss_vector=[]

for e in range(epochs):
    cumulative_loss_train = 0
    cumulative_loss_valid = 0
    for i, (data, label) in enumerate(data_iter_loader_train):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss_train = softmax_cross_entropy(output, label)    
        loss_train.backward()
        trainer.step(data.shape[0])
        cumulative_loss_train += nd.sum(loss_train).asscalar()
        print(cumulative_loss_train /42000,'   ***   ' )
        train_accuracy = evaluate_accuracy(data_iter_loader_train, net)
        print("Epoch %s , train_acc %s" %
          (e, train_accuracy))
    training_loss_vector.append(cumulative_loss_train)      

import matplotlib.pyplot as plt
x_axis = np.linspace(0 , epochs , len(training_loss_vector),endpoint=True )
plt.semilogy(x_axis , training_loss_vector)
plt.xlabel('epochs')
plt.ylabel('training_loss')
plt.show()

print ('Saving')
net.save_parameters("NN2_taskc.params")
print ('Saved')

test_accuracy = evaluate_accuracy(data_iter_loader_test , net)
print("TestAccuracy %s" % (test_accuracy))


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

    
dense_layer1=np.dot(a,train_images.T)+b.T
dense_layer2=np.dot(a1,dense_layer1)+b1.T
dense_layer3=np.dot(a2,dense_layer2)+b2.T

from sklearn import LogisticRegressor
log_reg=LogisticRegression().fit(np.transpose(dense_layer1),train_labels)
log_reg.score(test_images,test_labels)

