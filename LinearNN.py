import numpy as np
from copy import deepcopy
from MatrixGenerator import *
from numpy import linalg as LA
from scipy.linalg import eigh as largest_eigh

class LinearNN():

    def __init__(self, input, output, layers, name = 'LinearNN'):
        self.summary = Summary('LinearNN', name, input, output, layers)
        self.layers = []
        self.weight_product_back = {}
        self.weight_product = {}
        self.prediction = None
        self.X = None
        self.Y = None
        self.core = []
        self.loss = None
        self.learning_rate = None

    def set_weight(self,W):
        self.layers = [W]

    def forward(self, X, Y = None):
        self.X = X
        self.Y = Y
        self.core = []
        self.weight_product_back = {}
        self.weight_product = {}
        self.prediction = np.matmul(self.summary.a * self.weight_product_back_DP(), X)
        return self.prediction

    def weight_product_back_DP(self, i=0):
        if i in self.weight_product_back:
            return self.weight_product_back[i]
        if i == len(self.layers)-1:
            self.weight_product_back[i] = self.layers[i]
            return self.weight_product_back[i]
        self.weight_product_back[i] = np.matmul(self.weight_product_back_DP(i+1), self.layers[i])
        return self.weight_product_back[i]

    def weight_product_DP(self, i=None):
        if i == None:
            i = len(self.layers) - 1
        if i in self.weight_product:
            return self.weight_product[i]
        if i == 0:
            self.weight_product[i] = self.layers[i]
            return self.weight_product[i]
        self.weight_product[i] = np.matmul(self.layers[i], self.weight_product_DP(i-1))
        return self.weight_product[i]

    def backward(self):
        for i in range(len(self.layers)):
            self.layers[i] = self.layers[i] - self.lr() * self.gradient(i)

    def Loss(self):
        return (1/len(self.X[0])) * np.sum(np.subtract(self.prediction, self.Y) ** 2)

    def lr(self):
        if self.learning_rate == None:
            w, v = LA.eig(np.matmul(np.transpose(self.X),self.X))
            self.learning_rate =(1) * len(self.Y)/((2/len(self.X[0])) * float(max(w)) * len(self.layers))
            print(self.learning_rate)
        return self.learning_rate

    def gradient(self,i):
        if len(self.core) == 0:
            self.core = np.matmul(self.summary.a * np.subtract(self.prediction, self.Y), np.transpose(self.X))
        if i == 0:
            if len(self.layers) == 1:
                return (2/len(self.X[0])) * self.core
            return (2/len(self.X[0])) * self.summary.a * np.matmul(np.transpose(self.weight_product_back_DP(i + 1)),self.core)
        elif i < len(self.layers) -1:
            return (2/len(self.X[0])) * np.matmul(self.summary.a * np.matmul(np.transpose(self.weight_product_back_DP(i + 1)),self.core), np.transpose(self.weight_product_DP(i-1)))
        else:
            return (2/len(self.X[0])) * self.summary.a * np.matmul(self.core, np.transpose(self.weight_product_DP(i-1)))

    def compile(self, seed = None):
        if not seed == None:
            np.random.seed(seed)
        tmp = [self.summary.input] + self.summary.layers + [self.summary.output]
        x,y = 0, 0

        for d in tmp:
            x = y
            y = d
            if x and y:
                self.layers.append(np.random.rand(y,x))
                self.summary.parameter += x*y

    def __str__(self) -> str:
        return self.summary.__str__()

class Summary():

    def __init__(self, type, name, input, output, layers):
        self.layers = deepcopy(layers)
        self.type = type
        self.name = deepcopy(name)
        self.input = input
        self.output = output
        self.parameter = 0
        self.a = 1
        for layer in layers:
            self.a *= layer
        self.a *= output
        self.a = pow(self.a,-1/2)

    def __str__(self) -> str:
        return '\n'.join(['Type:',self.type,'' , 'Name:', self.name,'' , 'layers:',str(self.layers),'', 'a:', str(self.a), '', 'Parameter:', str(self.parameter)])




