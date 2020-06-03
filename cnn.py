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

def create_data():
	with tf.device(device_name):
	  c = ['blues', 'metal', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'pop', 'reggae', 'rock' ]
	  #c=["rock"]
	  y = np.zeros((len(c) * 100,1))
	  #print(c)
	  shapes = np.zeros((len(c) * 100, 432, 288, 1))
	  for j in range(len(c)):
	                  
	      print(j)
	          

	      for i in range(10):
	          
	          img = cv2.imread('drive/My Drive/CNNModel/genres/' + c[j]+   '/' + c[j] + '000' + str(0) + str(i) +'.png', cv2.IMREAD_GRAYSCALE) 
	          #print(c[j]+   '/' + c[j] + '000' + str(0) + str(i) +'.png')
	          #print(img.shape) (288, 432)
	          #img = img.reshape(1,124416)
	          img = img.reshape(432, 288, 1)
	          shapes[100*j + i]=  np.array(img)
	          y[100*j + i] = j
	      for i in range(10,100):
	          img = cv2.imread('drive/My Drive/CNNModel/genres/' + c[j]+   '/' + c[j] + '000' + str(i) +'.png', cv2.IMREAD_GRAYSCALE) 
	          #print(c[j]+   '/' + c[j] + '000' + str(i) +'.png')
	          #img = img.reshape(1,124416)
	          img = img.reshape(432, 288, 1)
	          shapes[100 *j  + i]=  np.array(img)
	          y[100*j + i] = j
	return shapes

	def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    with tf.device(device_name):
      m = X.shape[0]                  # number of training examples
      mini_batches = []
      np.random.seed(seed)
      
      # Step 1: Shuffle (X, Y)
      permutation = list(np.random.permutation(m))
      shuffled_X = X[permutation,:,:,:]
      shuffled_Y = Y[permutation,:]

      # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
      num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
      for k in range(0, num_complete_minibatches):
          mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
          mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
          mini_batch = (mini_batch_X, mini_batch_Y)
          mini_batches.append(mini_batch)
      
      # Handling the end case (last mini-batch < mini_batch_size)
      if m % mini_batch_size != 0:
          mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
          mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
          mini_batch = (mini_batch_X, mini_batch_Y)
          mini_batches.append(mini_batch)
      
      return mini_batches

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    with tf.device(device_name):
      # Retrieve the parameters from the dictionary "parameters" 
      W1 = parameters['W1']
      b1 = parameters['b1']
      W2 = parameters['W2']
      b2 = parameters['b2']
      W3 = parameters['W3']
      b3 = parameters['b3'] 
                                                            # Numpy Equivalents:
      Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
      A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
      Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
      A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
      Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
      
      return Z3

def predict(X, parameters):
    with tf.device(device_name):
      W1 = tf.convert_to_tensor(parameters["W1"])
      b1 = tf.convert_to_tensor(parameters["b1"])
      W2 = tf.convert_to_tensor(parameters["W2"])
      b2 = tf.convert_to_tensor(parameters["b2"])
      W3 = tf.convert_to_tensor(parameters["W3"])
      b3 = tf.convert_to_tensor(parameters["b3"])
      
      params = {"W1": W1,
                "b1": b1,
                "W2": W2,
                "b2": b2,
                "W3": W3,
                "b3": b3}
      
      x = tf.placeholder("float", [12288, 1])
      
      z3 = forward_propagation_for_predict(x, params)
      p = tf.argmax(z3)
      
      sess = tf.Session()
      prediction = sess.run(p, feed_dict = {x: X})
          
      return prediction

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """
    with tf.device(device_name):
      ### START CODE HERE ### (≈2 lines)
      X = tf.placeholder(dtype = 'float', shape = (None, n_H0, n_W0, n_C0), name = 'X')
      Y = tf.placeholder(dtype = 'float', shape = (None, n_y), name = 'Y')
      ### END CODE HERE ###
      
      return X, Y

"""
Consider changing shapes of these hard-coded weights
"""
def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 1, 8]
                        W2 : [2, 2, 8, 16]
    Note that we will hard code the shape values in the function to make the grading simpler.
    Normally, functions should take values as inputs rather than hard coding.
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    
    with tf.device(device_name):   
      ### START CODE HERE ### (approx. 2 lines of code)
      W1 = tf.get_variable("W1", [4, 4, 1, 8], initializer = tf.contrib.layers.xavier_initializer())
      W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer = tf.contrib.layers.xavier_initializer())
      ### END CODE HERE ###

      parameters = {"W1": W1,
                    "W2": W2}
      
      return parameters


