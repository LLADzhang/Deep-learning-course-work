import torch
from neural_network import NeuralNetwork

class AND:
    def __init__(self):
        self.and_nn = NeuralNetwork([2,1])
        self.layer = self.and_nn.getLayer(0)
    def __call__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        return self.forward()

    def forward(self):
        return self.and_nn.forward(torch.DoubleTensor([[self.x], [self.y]]))
