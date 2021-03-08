import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import torch
import torch.nn as nn


def npread(file, sep=',', header=None):
    df = pd.read_csv(file, sep=sep, header=header)
    arr = np.asarray(df)
    return arr

def parseData():
    directions = ['fx','fy','fz','mx','my','mz']

    X = np.empty((10*6+100*5*6,6))
    Y = np.empty((10*6+100*5*6,6))

    i = 0

    for direction1 in range(0,6):
        for direction2 in range(0,6):
            if direction1 == direction2:
                direction = direction1
                stroke_avg = []
                for resultingForce in range(0,6):
                    stroke_avg_tmp = []
                    cmd = []
                    for j in range(-5,5):
                        f_j = npread('ForceCharacterization/' + directions[direction] + '_{}.csv'.format(j))[:, resultingForce]
                        stroke_avg_tmp.append(np.mean(f_j[100:]))
                        cmd_tmp = [0,0,0,0,0,0]
                        cmd_tmp[direction] = 0.00001 * j
                        cmd.append(cmd_tmp)
                    stroke_avg.append(stroke_avg_tmp)
                Y[10*i:10*(i+1),:] = np.transpose(np.array(stroke_avg))
                X[10*i:10*(i+1),:] = np.array(cmd)
                i += 1
            else:
                for j1 in range(-5, 5):
                    stroke_avg = []
                    for resultingForce in range(0,6):
                        cmd = []
                        stroke_avg_tmp = []
                        for j2 in range(-5, 5):
                            f_j = npread('ForceCharacterization/'+directions[direction1]+'_'+directions[direction2]+'_'+str(j1)+'_'+str(j2)+'.csv')[:, resultingForce]
                            stroke_avg_tmp.append(np.mean(f_j[100:]))
                            cmd_tmp = [0,0,0,0,0,0]
                            cmd_tmp[direction1] = 0.00001*j1
                            cmd_tmp[direction2] = 0.00001*j2
                            cmd.append(cmd_tmp)
                        stroke_avg.append(stroke_avg_tmp)

                    Y[10*i:10*(i+1),:] = np.transpose(np.array(stroke_avg))
                    X[10*i:10*(i+1),:] = np.array(cmd)
                    i += 1
    np.save('CMD.npy',X)
    np.save('GenFM.npy',Y)

def readData():
    X = np.load('CMD.npy')
    Y = np.load('GenFM.npy')
    # X[:,2] -= 3.62
    # return torch.from_numpy(X).float(),torch.from_numpy(Y).float()
    return X, Y

class Neural_Network(nn.Module):
    def __init__(self, ):
        super(Neural_Network, self).__init__()
        # parameters
        self.inputSize = 6
        self.outputSize = 6
        self.hiddenSize1 = 10
        self.hiddenSize2 = 10
        self.hiddenSize3 = 10

        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize1)
        self.W2 = torch.randn(self.hiddenSize1, self.hiddenSize2)
        self.W3 = torch.randn(self.hiddenSize2, self.hiddenSize3)
        self.W4 = torch.randn(self.hiddenSize3, self.outputSize)

    def forward(self, X):
        self.z = torch.matmul(X, self.W1) # 3 X 3 ".dot" does not broadcast in PyTorch
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = torch.matmul(self.z2, self.W2)
        self.z4 = self.sigmoid(self.z3) # activation function
        self.z5 = torch.matmul(self.z4, self.W3)
        self.z6 = self.sigmoid(self.z5) # activation function
        self.z7 = torch.matmul(self.z6, self.W4)
        o = self.sigmoid(self.z7) # final activation function
        return o

    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s/100))

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s/100 * (1 - s)

    def backward(self, X, y, o):
        self.o_error = y - o # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o) # derivative of sig to error
        self.z6_error = torch.matmul(self.o_delta, torch.t(self.W4))
        self.z6_delta = self.z6_error * self.sigmoidPrime(self.z6)
        self.z4_error = torch.matmul(self.z6_delta, torch.t(self.W3))
        self.z4_delta = self.z4_error * self.sigmoidPrime(self.z4)
        self.z2_error = torch.matmul(self.z4_delta, torch.t(self.W2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += 0.1*torch.matmul(torch.t(X), self.z2_delta)
        self.W2 += 0.1*torch.matmul(torch.t(self.z2), self.z4_delta)
        self.W3 += 0.1*torch.matmul(torch.t(self.z4), self.z6_delta)
        self.W4 += 0.1*torch.matmul(torch.t(self.z6), self.o_delta)

    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)

    def saveWeights(self, model):
        # we will use the PyTorch internal storage functions
        torch.save(model, "NN")
        # you can reload model with all the weights and so forth with:
        # torch.load("NN")

    def predict(self):
        xPredicted = torch.tensor([0,0,1,0,0,0],dtype=torch.float)
        print ("Predicted data based on trained weights: ")
        print ("Input: \n" + str(xPredicted))
        print ("Output: \n" + str(self.forward(xPredicted)))

if __name__ == "__main__":
    # parseData()
    X, Y = readData()

    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(Y)

    train_perc = 0.9
    train = range(0,math.floor(X.shape[0]*train_perc))
    test = range(math.floor(X.shape[0]*train_perc),X.shape[0])
    x_train = torch.from_numpy(X[train,:]*1e5).float()
    x_test = torch.from_numpy(X[test,:]*1e5).float()
    y_train = torch.from_numpy(Y[train,:]).float()
    y_test = torch.from_numpy(Y[test,:]).float()

    NN = Neural_Network()
    Loss = []
    iter = 2000
    for i in range(iter):  # trains the NN 1,000 times
        loss = torch.mean((x_train - NN(y_train))**2).detach().item()
        print ("#" + str(i) + " Loss: " + str(loss))  # mean sum squared loss
        Loss.append(loss)
        NN.train(y_train, x_train)
    NN.saveWeights(NN)
    NN.predict()
    plt.plot(range(iter),Loss)
    plt.show()

