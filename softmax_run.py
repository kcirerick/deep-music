from softmax.py import *
#To be ran on google colab due to GPU dependencies
%tensorflow_version 1.x
import tensorflow as tf
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import cv2 

"""
The following code is Google Colab dependent
"""

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')

from google.colab import drive
drive.mount('/content/drive')

"""
"""
train_set_xx = shapes.T / 255
train_set_yy = y.T
from sklearn.utils import shuffle
train_set_xx, train_set_yy = shuffle(train_set_xx.T, train_set_yy.T)
train_set_yy = train_set_yy.astype(int)
train_set_x = (train_set_xx[0:950]).T
test_set_x = train_set_xx[950:].T
train_set_y = (train_set_yy[0:950]).T
test_set_y = (train_set_yy[950:]).T

Y_train = convert_to_one_hot(train_set_y, 10)
Y_test = convert_to_one_hot(test_set_y, 10)

"""Standard"""
parameters = model(train_set_x, Y_train, test_set_x, Y_test, learning_rate =10**-7, num_epochs = 500, minibatch_size = 32)

"""With Regularization"""
tr = np.array([])
ts = np.array([])
x= np.array([])
for i in range(4,8):
  x = np.append(x, i)
  a,b = model(train_set_x, Y_train, test_set_x, Y_test, learning_rate = 10**-i, num_epochs = 50, minibatch_size = 32)
  tr = np.append(tr, a)
  ts = np.append(ts, b)
plt.plot(x, tr, label = 'training')

plt.plot(x, ts, label = 'testing')
plt.xlabel('learning rate (log)')
plt.ylabel('Accuracy')
plt.legend()
plt.show()