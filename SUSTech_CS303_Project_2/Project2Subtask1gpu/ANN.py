from random import seed
import numpy as npp
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

def cross_entropy_loss(predict, label):
    lg = np.log(np.array(predict))
    loss = -np.mean(label * lg)
    return loss

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


class MyANN:
    def __init__(self, n_inputs, n_hidden, n_outputs, dropout_rate=0.0) -> None:
        self.network = []
        hidden_layer = [{'weights': [np.random.uniform(-0.5, 0.5) for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        self.network.append(hidden_layer)
        output_layer = [{'weights': [np.random.uniform(-0.5, 0.5) for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        self.network.append(output_layer)

    def forward(self, row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = neuron['weights'][-1]
                for i in range(len(neuron['weights']) - 1):
                    activation += neuron['weights'][i] * inputs[i]
                neuron['output'] = sigmoid(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs
    
    def backward(self, row, expected, output, learning_rate):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = []
            if i != len(self.network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])

            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * sigmoid_derivative(neuron['output'])
        
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
                
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += learning_rate * neuron['delta']

        
    def predict(self, test_data):
        return self.forward(test_data)

