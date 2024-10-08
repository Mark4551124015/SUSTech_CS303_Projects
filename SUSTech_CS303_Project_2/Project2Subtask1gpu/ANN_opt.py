from random import seed
# import numpy as npp
import cupy as np
from random import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import exp
from sklearn.metrics import accuracy_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)



class My_Opt:
    def __init__(self, alpha=0.5, beta1=0.9, beta2=0.999, epsilon=1e-8, steps=[], gamma=1.0):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.coe = 1.0
        self.steps  =steps
        self.gamma = gamma

    def get_learning_rate(self):
        lr = self.alpha * np.sqrt(1 - self.beta2 ** (self.t+1)) / (1 - self.beta1 ** (self.t+1))
        lr *= self.coe
        return lr
    
    def step(self):
        self.t += 1
        if self.t in self.steps:
            self.coe *= self.gamma


class MyANN_opt:
    def __init__(self, n_inputs, n_hidden, n_outputs, dropout_rate=0.0) -> None:


        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.dropout_rate = dropout_rate

        self.weights1 = np.random.randn(self.n_inputs, self.n_hidden)
        self.weights2 = np.random.randn(self.n_hidden, self.n_outputs)

    def forward(self, Input):
        self.z = np.dot(Input, self.weights1)
        self.z2 = sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.weights2)
        output = sigmoid(self.z3)
        return output
    
    def backward(self, Input, label, output, learning_rate):
        self.o_error = label - output
        self.o_delta = self.o_error * sigmoid_derivative(output)


        self.z2_error = self.o_delta.dot(self.weights2.T)
        self.z2_delta = self.z2_error * sigmoid_derivative(self.z2)


        self.weights1 += np.outer(Input, self.z2_delta) * learning_rate
        self.weights2 += np.outer(self.z2.T,self.o_delta) * learning_rate
        
        
    def predict(self, X):
        return self.forward(X)