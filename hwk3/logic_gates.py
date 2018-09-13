import torch
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt

class AND:
    def __init__(self):
        self.and_nn = NeuralNetwork([2,1])
        self.iterations = 1000   

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

            #print('\n\nITERATION', i)
            if self.and_nn.total_loss > 0.01:
                #print('dataset is', dataset)
                self.and_nn.forward(dataset)
                self.and_nn.backward(target)
                self.and_nn.updateParams(1)
            line, = plt.plot(i, self.and_nn.total_loss, 'r*')
        line.set_label("AND Gate")
        plt.xlabel('Iteration')
        plt.ylabel('Total Loss')
        plt.grid()


class OR:
    def __init__(self):
        self.or_nn = NeuralNetwork([2,1])
        self.iterations = 1000  
    
    def __call__(self, x, y):
        self.x, self.y = tuple(map(float, (x,y)))
        return bool(self.forward() > 0.5)

    def forward(self):
        return self.or_nn.forward(torch.DoubleTensor([[self.x,self.y]]))

    def train(self):
        dataset = torch.DoubleTensor([[0, 0],[0, 1],[1, 0],[1, 1]])
        target = torch.zeros(len(dataset), dtype=torch.double)
        target = torch.unsqueeze(target, 1)
        # target now has dimension 4X1

        for i in range(len(dataset)):
            target[i, :] = dataset[i, 0] or dataset[i, 1] 
            
        for i in range(self.iterations):
            #print('\n\nITERATION', i)
            if self.or_nn.total_loss > 0.01:
                #print('dataset is', dataset)
                self.or_nn.forward(dataset)
                self.or_nn.backward(target)
                self.or_nn.updateParams(1)
            line, = plt.plot(i, self.or_nn.total_loss, 'b*')

        line.set_label('OR Gate')

        plt.xlabel('Iteration')
        plt.ylabel('Total Loss')
        plt.grid()




class NOT:
    def __init__(self):
        self.not_nn = NeuralNetwork([1,1])
        self.iterations = 1000   
    
    def __call__(self, x):
        self.x = float(x)
        return bool(self.forward() > 0.5)

    def forward(self):
        return self.not_nn.forward(torch.DoubleTensor([[self.x]]))


    def train(self):
        dataset = torch.DoubleTensor([[0],[1]])
        target = torch.zeros(len(dataset), dtype=torch.double)
        target = torch.unsqueeze(target, 1)
        # target now has dimension 4X1

        for i in range(len(dataset)):
            target[i, :] = float(not dataset[i, 0])
            
        for i in range(self.iterations):
            #print('\n\nITERATION', i)
            if self.not_nn.total_loss > 0.01:
                # print('dataset is', dataset)
                self.not_nn.forward(dataset)
                self.not_nn.backward(target)
                self.not_nn.updateParams(1)
            line, = plt.plot(i, self.not_nn.total_loss, 'c*')
        line.set_label('NOT Gate')
        plt.xlabel('Iteration')
        plt.ylabel('Total Loss')
        plt.grid()


class XOR:
    def __init__(self):
        self.xor_nn = NeuralNetwork([2,2,1])
        self.iterations = 1000   

    def __call__(self, x, y):
        self.x, self.y = tuple(map(float, (x,y)))
        return bool(self.forward() > 0.5)

    def forward(self):
        return self.xor_nn.forward(torch.DoubleTensor([[self.x, self.y]]))

    def train(self):
        dataset = torch.DoubleTensor([[0, 0],[0, 1],[1, 0],[1, 1]])
        target = torch.zeros(len(dataset), dtype=torch.double)
        target = torch.unsqueeze(target, 1)
        # target now has dimension 4X1

        for i in range(len(dataset)):
            target[i, :] = float((dataset[i, 0] or dataset[i, 1]) and (not (dataset[i,0] and dataset[i, 1])))
            
        for i in range(self.iterations):
            #print('\n\nITERATION', i)
            if self.xor_nn.total_loss > 0.01:
              #  print('dataset is', dataset)
                self.xor_nn.forward(dataset)
                self.xor_nn.backward(target)
                self.xor_nn.updateParams(1)
            line, = plt.plot(i, self.xor_nn.total_loss, 'g*')
        line.set_label('XOR Gate')
        plt.xlabel('Iteration')
        plt.ylabel('Total Loss')
        plt.title('Loss vs Iterations for all gates')
        plt.grid()
        plt.legend()
        plt.savefig('loss.png')



