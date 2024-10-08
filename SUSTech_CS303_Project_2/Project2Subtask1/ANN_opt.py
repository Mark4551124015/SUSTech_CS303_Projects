from random import seed
import numpy as np
# import cupy as np
from random import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import exp
from sklearn.metrics import accuracy_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def soft_max(outputs):
    exp_output = np.exp(outputs)
    return exp_output/np.sum(exp_output)



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
    def __init__(self, n_inputs, n_hidden, n_outputs, dropout_rate=0.0, hidden_size=[8,6]) -> None:
        self.n_inputs = n_inputs
        self.n_hidden = len(hidden_size)
        self.n_outputs = n_outputs
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size

        self.layer_weight = []
        self.layer_bias = []
        last_size = n_inputs
        for size in hidden_size:
            w = np.random.uniform(-0.5, 0.5, size=(last_size,size))
            b = np.random.uniform(-0.5, 0.5, size=(size)) 
            self.layer_weight.append(w)
            self.layer_bias.append(b)
            last_size = size
        w = np.random.uniform(-0.5, 0.5, size=(last_size,n_outputs))
        b = np.random.uniform(-0.5, 0.5, size=(n_outputs)) 
        self.out_weights = (w)
        self.out_bias = (b)


    def forward(self, Input, training=True):
        self.layer_outputs = [None] * self.n_hidden 
        self.layer_inputs = [None] * self.n_hidden
        self.layer_masks = [None] * self.n_hidden

        last_out = Input
        for i in range(self.n_hidden):
            self.layer_inputs[i] = np.dot(last_out, self.layer_weight[i]) + self.layer_bias[i]
            self.layer_outputs[i] = sigmoid(self.layer_inputs[i])
            if training and self.dropout_rate > 0:
                self.layer_masks[i] = np.random.binomial(1, 1 - self.dropout_rate, size=self.layer_outputs[i].shape) / (1 - self.dropout_rate)
                self.layer_outputs[i] *= self.layer_masks[i]
            last_out = self.layer_outputs[i]

        self.out_input = np.dot(last_out, self.out_weights) + self.out_bias
        output = soft_max(self.out_input)
        return output
    
    def backward(self, Input, label, output, learning_rate, training=True):
        self.layer_error = [None] * self.n_hidden
        self.layer_delta = [None] * self.n_hidden

        self.out_error = label - output
        self.out_delta = self.out_error * sigmoid_derivative(output)

        error = self.out_delta.dot(self.out_weights.T)
        for i in reversed(range(self.n_hidden)):
            self.layer_delta[i] = error * sigmoid_derivative(self.layer_outputs[i])
            error = self.layer_delta[i].dot(self.layer_weight[i].T)
            if training and self.dropout_rate > 0:
                self.layer_delta[i] *= self.layer_masks[i]


        last_out = Input
        for i in range(self.n_hidden):
            self.layer_weight[i] += np.outer(last_out, self.layer_delta[i]) * learning_rate
            self.layer_bias[i] += learning_rate * np.sum(self.layer_delta[i], axis=0)
            last_out = self.layer_outputs[i]
        self.out_weights += np.outer(last_out,self.out_delta) * learning_rate
        self.out_bias += learning_rate * np.sum(self.out_delta, axis=0)
        
    def predict(self, X):
        return self.forward(X)
    

