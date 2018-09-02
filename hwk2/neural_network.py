import torch
import numpy as np
from math import sqrt, exp

class NeuralNetwork:
    def __init__(self, layers):
        # layers is a list of layer sizes
        if type(layers) != list:
            raise TypeError('Input is not a list')

        self.layers = layers
        self.theta = {}
        # n layers neural network has n-1 weight matrices
        for i in range(len(layers) - 1):
            # the diemension includes one position for biasi
            size = (self.layers[i] + 1, self.layers[i+1])
            self.theta[i] = torch.normal(
                                torch.zeros(size[0], size[1]),
                                1/sqrt(self.layers[i]),  
                            )
   
    
    def getLayer(self, layer):
        if layer not in self.theta.keys():
            raise ValueError('Layer index not exists')
        # layer is an integer for the layer index
        # return the corresponding theta matric from that layer to layer + 1
        return self.theta[layer]

        
    def forward(self, nn_input):
        # the one iteration forward function
        def sigmoid(i):
            if type(i) != torch.DoubleTensor:
                raise TypeError('Input of sigmoid is not DoubleTensor')
            return torch.pow(exp(1), i)
       if type(i) != torch.DoubleTensor:
           raise TypeError('Input of forward is not DoubleTensor')
        bias = torch.ones(1, nn_input.size()[1], dtype=torch.double)
        for i in self.theta.keys():
            operation_input =  torch.cat((nn_input, bias), 0)
            operation_input = sigmoid(torch.mm(torch.t(self.theta[i]), operation_input))
        return operation_input
        
