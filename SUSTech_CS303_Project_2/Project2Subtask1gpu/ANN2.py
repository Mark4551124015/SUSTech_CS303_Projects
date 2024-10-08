
# import numpy as np
import cupy as np

class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden, n_outputs, dropout_rate=0.0):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.dropout_rate = dropout_rate
        
        # Initialize weights randomly
        self.weights1 = np.random.randn(self.n_inputs, self.n_hidden)
        self.weights2 = np.random.randn(self.n_hidden, self.n_outputs)
        
    def forward(self, X, training=True):
        # Forward propagation through the network
        self.z = np.dot(X, self.weights1)
        self.z2 = self.sigmoid(self.z)
        # if training:
        #     # Apply dropout to the hidden layer during training
        #     self.dropout_mask = np.random.binomial(1, 1-self.dropout_rate, size=self.z2.shape) / (1-self.dropout_rate)
        #     self.z2 *= self.dropout_mask
        self.z3 = np.dot(self.z2, self.weights2)
        o = self.sigmoid(self.z3)
        return o
    
    def sigmoid(self, s):
        # Activation function
        return 1/(1+np.exp(-s))
    
    def sigmoid_prime(self, s):
        # Derivative of sigmoid function
        return s * (1 - s)
    
    def backward(self, X, y, o, learning_rate):
        # Backward propagation through the network
        self.o_error = y - o
        self.o_delta = self.o_error * self.sigmoid_prime(o)
        
        self.z2_error = self.o_delta.dot(self.weights2.T)
        self.z2_delta = self.z2_error * self.sigmoid_prime(self.z2)

        self.weights1 += np.outer(X, self.z2_delta) * learning_rate
        self.weights2 += np.outer(self.z2.T,self.o_delta) * learning_rate
        
    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)
        
    def predict(self, X):
        # Predict the output for the given input
        return self.forward(X, training=False)
