import torch
from math import sqrt, exp

class NeuralNetwork:
    def __init__(self, layers):
        # layers is a list of layer sizes
        if type(layers) != list:
            raise TypeError('Input is not a list')

        self.layers = layers
        self.theta = {}
        self.dE_dTheta = {}
        self.a = {} # the result after applying the sigmoid functino
        self.z = {} # result after weight matrix multiplies the activation
        # self.L is the index of the output layer
        self.L = len(layers) - 1

        # n layers neural network has n-1 weight matrices
        for i in range(len(self.layers) - 1):
            # the diemension includes one position for biasi
            size = (self.layers[i] + 1, self.layers[i+1])
            self.theta[i] = torch.normal(
                                torch.zeros(size[0], size[1]),
                                1/sqrt(self.layers[i])
                            ).type(torch.DoubleTensor)
        self.total_loss = 1 


    def getLayer(self, layer):
        
        if layer not in self.theta.keys():
            raise ValueError('Layer index not exists')
        # layer is an integer for the layer index
        # return the corresponding theta matric from that layer to layer + 1
        return self.theta[layer]

        
    def forward(self, nn_input):
        # nn_input is mXn where m is the number of samples
        # n is the number of neurons in each sample
        print('original input', nn_input)
        # the one iteration forward function
        def sigmoid(i):
            if str(i.type()) != 'torch.DoubleTensor':
                raise TypeError('Input of sigmoid is not DoubleTensor')
            return 1 / (1 + torch.pow(exp(1), -i))

        if str(nn_input.type()) != 'torch.DoubleTensor':
            raise TypeError('Input of forward is not DoubleTensor')
        si = [1, nn_input.size()[0]]
        print('si', si)

        bias = torch.ones(si, dtype=torch.double)
        print('bias', bias)
        operation_input = nn_input.t()
        # operation_input has nxm dim
        self.a[0] = nn_input.t()
        print('a[0]', self.a[0])

        for i in self.theta.keys():
            self.a[i] =  torch.cat((operation_input, bias), 0)
            print('cat input', self.a[i])
            theta = torch.t(self.theta[i])
            print('theta', theta)
            self.z[i + 1] = torch.mm(theta, self.a[i])
            print('z', self.z[i+1])
            self.a[i + 1] = sigmoid(self.z[i + 1])
            print('a', self.a[i+1])
            bias = torch.ones([1] + list(self.a[i].size()[1:]), dtype=torch.double)
        print('return from forward', self.a[self.L].t())
        return self.a[self.L].t() 


    def backward(self, target, loss='MSE'):
        print('target', target)
        if loss == 'MSE':
            # step 1 calculate the loss function
            self.total_loss = (self.a[self.L] - target).pow(2).sum() / 2 / len(target)
            print('output activation:', self.a[self.L])
            print('total loss', self.total_loss)
            delta = torch.mul((self.a[self.L] - target), torch.mul(self.a[self.L], (1 - self.a[self.L])))
            print('delta', delta)
            
            for i in range(self.L - 1, -1, -1):
                    
                # from the layer before the output
                self.dE_dTheta = torch.mm(self.a[i], delta.t())
                delta = torch.mul(torch.mm(self.theta[i].t(),delta), torch.mul(self.a[i], (1 - self.a[i])))
        elif loss == 'CE':
            pass

        else:
            print('unrecognized error functino')
    def updateParam(self, rate):
        for i in range(len(self.theta)):
            self.theta[i] = self.theta[i] - torch.mul(self.dE_dTheta[i], rate)
