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
train_image,train_label=Dataset.load_data()
test_images,test_labels=Dataset.load_data_test()
a=int(60000*0.7)
b=int(60000*0.3)
train_labels=train_label[:a]
train_labels_validation=train_label[a:60000]
train_images=train_image[:a,:]
train_images_validation=train_image[a:60000,:]
Batch_size=128
data_iter_train = mx.io.NDArrayIter(data=train_images, label=train_labels, batch_size=Batch_size)
data_iter_validation = mx.io.NDArrayIter(data=train_images_validation, label=train_labels_validation, batch_size=Batch_size)
data_iter_test = mx.io.NDArrayIter(data=test_images, label=test_labels, batch_size=Batch_size)

class DataIterLoader():
    def __init__(self, data_iter):
        self.data_iter = data_iter

    def __iter__(self):
        self.data_iter.reset()
        return self

    def __next__(self):
        batch = self.data_iter.__next__()
        assert len(batch.data) == len(batch.label) == 1
        data = batch.data[0]
        label = batch.label[0]
        return data, label

    def next(self):
        return self.__next__() # for Python 2
    
data_iter_loader_train = DataIterLoader(data_iter_train)
data_iter_loader_valid = DataIterLoader(data_iter_validation)
data_iter_loader_test = DataIterLoader(data_iter_test)

from multiprocessing import cpu_count
CPU_COUNT = cpu_count()

ctx = mx.gpu() 
data_ctx = ctx
model_ctx = ctx
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(1024, activation="relu"))
    net.add(gluon.nn.BatchNorm())
    net.add(gluon.nn.Dense(512, activation="relu"))
    net.add(gluon.nn.BatchNorm())
    net.add(gluon.nn.Dense(256, activation="relu"))
    net.add(gluon.nn.BatchNorm())
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
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': .001})
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
    for j, (data, label) in enumerate(data_iter_loader_valid):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss_valid = softmax_cross_entropy(output, label)
        cumulative_loss_valid += nd.sum(loss_valid).asscalar()
        print(cumulative_loss_train /42000,'   ***   '  , '***' , cumulative_loss_valid/18000)
        validation_accuracy=evaluate_accuracy(data_iter_loader_valid , net)
        train_accuracy = evaluate_accuracy(data_iter_loader_train, net)
        print("Epoch %s , train_acc %s, validation_acc %s" %
          (e, train_accuracy, validation_accuracy))
    training_loss_vector.append(cumulative_loss_train)
    validation_loss_vector.append(cumulative_loss_valid)
      

import matplotlib.pyplot as plt
x_axis = np.linspace(0 , epochs , len(training_loss_vector),endpoint=True )
plt.semilogy(x_axis , training_loss_vector)
plt.xlabel('epochs')
plt.ylabel('training_loss')
plt.show()

x_axis = np.linspace(0 , epochs , len(validation_loss_vector) )
plt.semilogy(x_axis , validation_loss_vector)
plt.xlabel('epochs')
plt.ylabel('training_loss')
plt.show()

print ('Saving')
net.save_parameters("NN2_batch_norm.params")
print ('Saved')


test_accuracy = evaluate_accuracy(data_iter_loader_test , net)
print("TestAccuracy %s" % (test_accuracy))
"""
epochs=50
train acc=95.8
valid acc=88.2
test acc= 87.5
"""



