import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import torch 

class MultiLayerPerceptron:
  def __init__(self,eta=0.1,gamma=0.1,stepsize=200,threshold=0.08,test_interval=10,max_epoch=3000,hidden_size=100):
    self.eta = eta
    self.gamma = gamma
    self.stepsize = stepsize
    self.threshold = threshold
    self.test_interval = test_interval
    self.max_epoch = max_epoch
    self.hidden_size = hidden_size 

    #fetch the training/validation data
    self.x_train = torch.from_numpy(pd.read_csv('data/training_set.csv', header=None).values)
    self.y_train = torch.from_numpy(pd.read_csv('data/training_labels_bin.csv', header=None).values)
    self.x_val = torch.from_numpy(pd.read_csv('data/validation_set.csv', header=None).values)
    self.y_val = torch.from_numpy(pd.read_csv('data/validation_labels_bin.csv', header=None).values)

    #fetch important variable from the data
    num_feats = self.x_train.shape[1]
    n_out = self.y_train.shape[1]

    #initialize the weights and biases randomly
    self.weights = [torch.randn(num_feats, self.hidden_size), 
                    torch.randn(self.hidden_size, self.hidden_size),
                    torch.randn(self.hidden_size, n_out)]
    self.biases = [torch.randn(self.hidden_size),
                   torch.randn(self.hidden_size),
                   torch.randn(n_out)]

  def forward(self, x):
    #store the outputs of each layer
    self.a = [x]

    #pass through every layer, executing the matrix multiplication then sigmoid activ fct 
    for i in range(len(self.weights)):
      z = torch.mm(self.a[-1], self.weights[i]) + self.biases[i]
      self.a.append(torch.sigmoid(z))

    return self.a[-1]
  
  def loss(self, y_hat, y):
    return torch.sum((y_hat - y)**2)

  def backward(self, x, y, y_hat):
    #compute gradient of loss wrt y_hat
    dy_hat = 2 * (y_hat - y)

    #init lists to store gradients
    dweights = [None] * len(self.weights)
    dbiases = [None] * len(self.biases)

    #backprop through the layers
    for i in range(len(self.weights)-1, -1, -1):
      dweights[i] = torch.mm(self.a[i].T, dy_hat)
      dbiases[i] = torch.sum(dy_hat, axis=0)

      if i != 0: #skip for this for the first layer, since it's the sigmoid derivative
        dy_hat = torch.mm(dy_hat, self.weights[i].T) * self.a[i] * (1-self.a[i])

    return dweights, dbiases
    
  def train(self, x, y):
    total_loss = 0
    
    #iterate over each training point 
    for i in range(len(x)):
      y_hat = self.forward(x[i])
      loss = self.loss(y_hat, y[i])
      total_loss += loss.item()

      dweights, dbiases = self.backward(x[i], y[i], y_hat)

      #update weights and biases
      for j in range(len(self.weights)):
        self.weights[j] -= self.eta * dweights[j]
        self.biases[j] -= self.eta * dbiases[j]
      
    return total_loss / len(x)

