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

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Note that for simplicity and grading purposes, we'll hard-code some values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    with tf.device(device_name):
      # Retrieve the parameters from the dictionary "parameters" 
      W1 = parameters['W1']
      W2 = parameters['W2']
      #print('start')
      ### START CODE HERE ###
      # CONV2D: stride of 1, padding 'SAME'
      #print(X.shape)
      Z1 = tf.nn.conv2d(X, W1, strides = [1, 1, 1, 1], padding = 'SAME')
      #print('z1')
      # RELU
      A1 = tf.nn.relu(Z1)
      #print('A1')
      # MAXPOOL: window 8x8, stride 8, padding 'SAME'
      P1 = tf.nn.max_pool(A1, ksize = [1, 8, 8, 1], strides = [1, 8, 8, 1], padding = 'SAME')
      #print('P1')
      # CONV2D: filters W2, stride 1, padding 'SAME'
      Z2 = tf.nn.conv2d(P1, W2, strides = [1, 1, 1, 1], padding = 'SAME')
      #print('Z2')
      # RELU
      A2 = tf.nn.relu(Z2)
      #print('A2')
      # MAXPOOL: window 4x4, stride 4, padding 'SAME'
      P2 = tf.nn.max_pool(A2, ksize = [1, 4, 4, 1], strides = [1, 4, 4, 1], padding = 'SAME')
      #print('P2')
      # FLATTEN
      F = tf.contrib.layers.flatten(P2)
      #print('F')
      # FULLY-CONNECTED without non-linear activation function (not not call softmax).
      # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
      Z3 = tf.contrib.layers.fully_connected(F, 10, activation_fn = None)
      #print('Z3')
      ### END CODE HERE ###

      return Z3

def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (number of examples, 6)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
      ### START CODE HERE ### (1 line of code)
    with tf.device(device_name):
      cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
      ### END CODE HERE ###
    
      return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 50, minibatch_size = 64, print_cost = True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    with tf.device(device_name):
      ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
      seed = 3                                          # to keep results consistent (numpy seed)
      (m, n_H0, n_W0, n_C0) = X_train.shape           
      n_y = Y_train.shape[1]                            
      costs = []                                        # To keep track of the cost
      
      #print(n_C0)
      # Create Placeholders of the correct shape
      ### START CODE HERE ### (1 line)
      X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
      #print('created')
      ### END CODE HERE ###

      # Initialize parameters
      ### START CODE HERE ### (1 line)
      parameters = initialize_parameters()
      #print('initialized')
      ### END CODE HERE ###
      
      # Forward propagation: Build the forward propagation in the tensorflow graph
      ### START CODE HERE ### (1 line)
      Z3 = forward_propagation(X, parameters)
      #print('forward')
      ### END CODE HERE ###
      
      # Cost function: Add cost function to tensorflow graph
      ### START CODE HERE ### (1 line)
      cost = compute_cost(Z3, Y)
      #print('costed')
      ### END CODE HERE ###
      
      # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
      ### START CODE HERE ### (1 line)
      optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
      #print('optimized')
      ### END CODE HERE ###
      
      # Initialize all the variables globally
      init = tf.global_variables_initializer()
      #print('global init')
      
      # Start the session to compute the tensorflow graph
      with tf.Session() as sess:
          
          # Run the initialization
          sess.run(init)
          #print('sessinit')
          
          # Do the training loop
          for epoch in range(num_epochs):

              minibatch_cost = 0.
              num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
              seed = seed + 1
              minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

              for minibatch in minibatches:

                  # Select a minibatch
                  (minibatch_X, minibatch_Y) = minibatch
                  """
                  # IMPORTANT: The line that runs the graph on a minibatch.
                  # Run the session to execute the optimizer and the cost.
                  # The feedict should contain a minibatch for (X,Y).
                  """
                  ### START CODE HERE ### (1 line)
                  _ , temp_cost =  sess.run(fetches=[optimizer, cost], 
                                            feed_dict={X: minibatch_X,
                                                      Y: minibatch_Y})
                  ### END CODE HERE ###
                  
                  minibatch_cost += temp_cost / num_minibatches
                  

              # Print the cost every epoch
              if print_cost == True and epoch % 5 == 0:
                  print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
                  predict_op = tf.argmax(Z3, 1)
                  correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
                  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                  
                  test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
                  print(test_accuracy)
              if print_cost == True and epoch % 1 == 0:
                  costs.append(minibatch_cost)
          
          
          # plot the cost
          plt.plot(np.squeeze(costs))
          plt.ylabel('cost')
          plt.xlabel('iterations (per tens)')
          plt.title("Learning rate =" + str(learning_rate))
          plt.show()

          # Calculate the correct predictions
          tf.print(Z3)
          print('on god')
          predict_op = tf.argmax(Z3, 1)
          correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
          
          # Calculate accuracy on the test set
          accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
          train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
          test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
          print("Train Accuracy:", train_accuracy)
          print("Test Accuracy:", test_accuracy)
                  
          return  train_accuracy, test_accuracy, parameters

