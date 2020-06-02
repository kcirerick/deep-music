#CNN Model to be ran on Google Colab,
#Due to dependency on GPU Access
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
%tensorflow_version 1.x
import tensorflow as tf

from tensorflow.python.framework import ops
import cv2
device_name = tf.test.gpu_device_name()
from cnn.py import *
from sklearn.utils import shuffle

shapes = create_data() #May need to chane the path in cnn.py to run
train_set_xx = shapes.T / 255
train_set_yy = y.T
train_set_xx, train_set_yy = shuffle(train_set_xx.T, train_set_yy.T, random_state=0)
train_set_yy = train_set_yy.astype(int)
train_set_x = (train_set_xx[0:950])
test_set_x = train_set_xx[950:]
train_set_y_orig = (train_set_yy[0:950]).T
test_set_y_orig = (train_set_yy[950:]).T
train_set_y = convert_to_one_hot(train_set_y_orig, 10).T
test_set_y = convert_to_one_hot(test_set_y_orig, 10).T

#Runs a few iterations of the cnn model on grayscale spectograms
#searching through different learning rates.
for i in range(8,12):
  conv_layers = {}
  _, _, parameters = model(train_set_x, train_set_y, test_set_x, test_set_y, learning_rate= (0.5**i), minibatch_size = 32, num_epochs=200)
