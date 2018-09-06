import torch
from neural_network import NeuralNetwork

class AND:
    def __init__(self):
        self.and_nn = NeuralNetwork([2,1])
        self.layer = self.and_nn.getLayer(0)
        self.layer[:,:] = 0.0
        self.layer += torch.DoubleTensor([[10],[10], [-15]])
    
    def __call__(self, x, y):
        self.x, self.y = tuple(map(float, (x,y)))
        return bool(self.forward() > 0.5)

    def forward(self):
        return self.and_nn.forward(torch.DoubleTensor([[self.x], [self.y]]))

class OR:
    def __init__(self):
        self.or_nn = NeuralNetwork([2,1])
        self.layer = self.or_nn.getLayer(0)
        self.layer[:,:] = 0.0
        self.layer += torch.DoubleTensor([[15],[15], [-10]])
    
    def __call__(self, x, y):
        self.x, self.y = tuple(map(float, (x,y)))
        return bool(self.forward() > 0.5)

    def forward(self):
        return self.or_nn.forward(torch.DoubleTensor([[self.x], [self.y]]))

class NOT:
    def __init__(self):
        self.not_nn = NeuralNetwork([1,1])
        self.layer = self.not_nn.getLayer(0)
        self.layer[:,:] = 0.0
        self.layer += torch.DoubleTensor([[-20], [10]])
    
    def __call__(self, x):
        self.x = float(x)
        return bool(self.forward() > 0.5)

    def forward(self):
        return self.not_nn.forward(torch.DoubleTensor([[self.x]]))

class XOR:
    def __init__(self):
        self.xor_nn = NeuralNetwork([2,2,1])
        self.layer0 = self.xor_nn.getLayer(0)
        self.layer1 = self.xor_nn.getLayer(1)

        self.layer0[:,:] = 0.0 
        self.layer0 += torch.DoubleTensor([[15, -10], [15, -10], [-10, 15]])
        self.layer1[:, :] = 0.0
        self.layer1 += torch.DoubleTensor([[10],[10],[-15]])

    def __call__(self, x, y):
        self.x, self.y = tuple(map(float, (x,y)))
        return bool(self.forward() > 0.5)

    def forward(self):
        return self.xor_nn.forward(torch.DoubleTensor([[self.x], [self.y]]))
