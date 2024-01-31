import numpy as np
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
    self.num_feats = self.x_train.shape[1]
    self.n_out = self.y_train.shape[1]

    #initialize the weights and biases randomly
    self.weights = [torch.randn(self.num_feats, self.hidden_size).double(), 
            torch.randn(self.hidden_size, self.hidden_size).double(),
            torch.randn(self.hidden_size, self.n_out).double()]
    self.biases = [torch.randn(self.hidden_size).double(),
            torch.randn(self.hidden_size).double(),
            torch.randn(self.n_out).double()]

  def forward(self, x):
    #store the outputs of each layer
    self.a = [x]

    #first hidden layer
    z1 = torch.mm(self.a[-1].unsqueeze(0), self.weights[0]) + self.biases[0]
    self.a.append(torch.sigmoid(z1))

    #second hidden layer
    z2 = torch.mm(self.a[-1], self.weights[1]) + self.biases[1]
    self.a.append(torch.sigmoid(z2))

    #output layer
    z3 = torch.mm(self.a[-1], self.weights[2]) + self.biases[2]
    self.a.append(torch.sigmoid(z3))

    return self.a[-1]
  
  def loss(self, y_hat, y):
    return torch.sum((y_hat - y)**2)

  def backward(self, x, y, y_hat):
    #compute gradient of loss wrt y_hat
    dy_hat = 2 * (y_hat - y)

    #init lists to store gradients
    dweights = [None] * len(self.weights)
    dbiases = [None] * len(self.biases)

    #output layer
    dweights[2] = torch.mm(self.a[2].T, dy_hat)
    dbiases[2] = torch.sum(dy_hat, axis=0)
    dy_hat = torch.mm(dy_hat, self.weights[2].T) * self.a[2] * (1-self.a[2])

    #second hidden layer
    dweights[1] = torch.mm(self.a[1].T, dy_hat)
    dbiases[1] = torch.sum(dy_hat, axis=0)
    dy_hat = torch.mm(dy_hat, self.weights[1].T) * self.a[1] * (1-self.a[1])

    #first hidden layer
    dweights[0] = torch.mm(self.a[0].unsqueeze(0).T, dy_hat)
    dbiases[0] = torch.sum(dy_hat, axis=0)

    return dweights, dbiases

  def test(self):
    val_sse = 0

    for i in range(len(self.x_val)):
      x_in = self.x_val[i]#.reshape(1, self.num_feats)
      y = self.y_val[i]#.reshape(1, self.n_out)

      #predict
      y_hat = self.forward(x_in)

      #compute loss
      val_loss = self.loss(y_hat, y)
      val_sse += val_loss

    #compute mean squared error
    val_mse = val_sse / len(self.x_val)

    return val_mse 
    
  def train(self):
    train_mses = []
    val_mses = []
    x = self.x_train
    y = self.y_train
    
    for epoch in range(0, self.max_epoch):
      #shuffle data
      order = np.random.permutation(len(x))

      sse = 0
      for n in range(0,len(x)):
        idx = order[n]
        
        #get a sample (batch size=1)
        x_in = self.x_train[idx]#.reshape((self.num_feats,1))
        y = self.y_train[idx]#.reshape((self.n_out, 1))
        
        #do the forward pass here
        y_hat = self.forward(x_in)
        
        #compute loss
        loss = self.loss(y_hat, self.y_train[idx])
        sse += loss

        #compute gradients
        dweights, dbiases = self.backward(x_in, y, y_hat)

        #update weights and biases
        for j in range(len(self.weights)):
          self.weights[j] -= self.eta * dweights[j]
          self.biases[j] -= self.eta * dbiases[j]
          
      train_mse = sse / len(self.x_train)
      train_mses.append(train_mse)

      if epoch % self.test_interval == 0: 
        val_mse = self.test()
        val_mses.append(val_mse)
        print(f'Epoch {epoch}: train MSE = {train_mse}, validation MSE = {val_mse}')
        # if termination condition is satisfied, exit
        if val_mse < self.threshold:
          print("val_mse < threshold")
          break

      #MIGHT IMPLEMENT LATER
      # if epoch % self.stepsize == 0 and epoch != 0:
      #     self.eta = self.eta * self.gamma
      #     print('Changed learning rate to lr=' + str(self.eta))
          
    plt.plot(train_mses, label='Training MSE')
    plt.plot(np.arange(0, len(train_mses), self.test_interval), val_mses, label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
    return train_mse
      

mlp = MultiLayerPerceptron()
loss = mlp.train()
print(loss)