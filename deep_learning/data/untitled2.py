# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 15:57:06 2019

@author: praneet jaiin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 17:41:31 2019

@author: praneet jaiin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 14:28:12 2019

@author: praneet jaiin
"""

from zipfile import ZipFile
import numpy as np

import mxnet as mx
from mxnet import nd , autograd , gluon
import numpy as np

class DataLoader(object):
    def __init__(self):
        DIR = '\ass_2'
        pass
    
    # Returns images and labels corresponding for training and testing. Default mode is train. 
    # For retrieving test data pass mode as 'test' in function call.
    def load_data(self, mode = 'train'):
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = '../ass_2/' + label_filename + '.zip'
        image_zip = '../ass_2/' + image_filename + '.zip'
        with ZipFile(label_zip, 'r') as lblzip:
            labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
        with ZipFile(image_zip, 'r') as imgzip:
            images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        return images, labels
    def load_data_test(self, mode = 'test'):
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = '../ass_2/' + label_filename + '.zip'
        image_zip = '../ass_2/' + image_filename + '.zip'
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
data_iter = mx.io.NDArrayIter(data=train_images, label=train_labels, batch_size=256)
data_iter_test = mx.io.NDArrayIter(data=test_images, label=test_labels, batch_size=256)
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
    
data_iter_loader = DataIterLoader(data_iter)
data_iter_test_loader = DataIterLoader(data_iter_test)

from multiprocessing import cpu_count
CPU_COUNT = cpu_count()

ctx = mx.gpu() 
data_ctx = ctx
model_ctx = ctx

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(1024, activation="relu"))
    net.add(gluon.nn.Dropout(.1))
    net.add(gluon.nn.BatchNorm())
    net.add(gluon.nn.Dense(512, activation="relu"))
    net.add(gluon.nn.Dropout(.4))
    net.add(gluon.nn.Dense(256, activation="relu"))
    net.add(gluon.nn.Dropout(.6))
    net.add(gluon.nn.Dense(10 , activation='softrelu'))
net.collect_params().initialize(mx.init.Normal(sigma=.01), ctx=model_ctx)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iter_loader):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


epochs = 250
smoothing_constant = .001
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})
for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(data_iter_loader):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        cumulative_loss += nd.sum(loss).asscalar()
        print(cumulative_loss /60000,'   ***   ' , e)
        test_accuracy = evaluate_accuracy(data_iter_test_loader, net)
        train_accuracy = evaluate_accuracy(data_iter_loader, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
          (e, cumulative_loss/60000, train_accuracy, test_accuracy))




