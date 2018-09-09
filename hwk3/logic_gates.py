import torch
from neural_network import NeuralNetwork

class AND:
    def __init__(self):
        self.and_nn = NeuralNetwork([2,1])
        self.iterations = 100    

    def __call__(self, x, y):
        self.x, self.y = tuple(map(float, (x,y)))
        return bool(self.forward() > 0.5)

    def forward(self):
        return self.and_nn.forward(torch.DoubleTensor([[self.x, self.y]]))


    def train(self):
        dataset = torch.DoubleTensor([[0, 0],[0, 1],[1, 0],[1, 1]])
        target = torch.zeros(len(dataset), dtype=torch.double)
        target = torch.unsqueeze(target, 1)
        # target now has dimension 4X1

        for i in range(len(dataset)):
            target[i, :] = dataset[i, 0] and dataset[i, 1] 
            
        for i in range(self.iterations):
            print('\n\nITERATION', i)
            if self.and_nn.total_loss > 0.01:
                print('dataset is', dataset)
                self.and_nn.forward(dataset)
                self.and_nn.backward(target)
                self.and_nn.updateParams(1)


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
