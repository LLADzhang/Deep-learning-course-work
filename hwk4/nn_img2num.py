from pprint import pprint as pp
from neural_network import NeuralNetwork 
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from time import time
from torch.autograd import Variable

class NNImg2Num:
    def __init__(self):
        self.train_batch_size = 60
        self.epoch = 20
        self.labels = 10
        self.rate = 0.1 
        self.input_size = 28 * 28
        self.test_batch_size = 10 * self.train_batch_size
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', 
                train=True, 
                download=True, 
                transform=transforms.Compose([transforms.ToTensor()])),
                batch_size=self.test_batch_size, shuffle=True)

        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', 
                train=True, 
                download=True, 
                transform=transforms.Compose([transforms.ToTensor()])),
                batch_size=self.train_batch_size, shuffle=True)
        
        # input image is 28 * 28 so convert to 1D matrix
        # output labels are 10 [0 - 9]
        self.nn = nn.Sequential(
                nn.Linear(self.input_size, 512), nn.Sigmoid(),
                nn.Linear(512, 256), nn.Sigmoid(),
                nn.Linear(256, 64), nn.Sigmoid(),
                nn.Linear(64, self.labels), nn.Sigmoid(),
                )

    def train(self):
        optimizer = torch.optim.SGD(self.nn.parameters(), lr=self.rate, momentum=0.9)
        loss_function = nn.MSELoss()
        print('training')
        
        def onehot_training(target, batch_size):
                output = torch.zeros(batch_size, self.labels)
                for i in range(batch_size):
                    output[i][int(target[i])] = 1.0
                return output

        def training():
            loss = 0
            for batch_id, (data, target) in enumerate(self.train_loader):
                #print('batch {} out of {} batches'.format(batch_id, len(self.train_loader.dataset)/ self.train_batch_size))
                #print(data.view(self.train_batch_size, self.input_size).type(torch.DoubleTensor).type())
                #print(target.size())
                # data.view change the dimension of input to use forward function
                optimizer.zero_grad()
                forward_pass_output = self.nn(data.view(self.train_batch_size, self.input_size))
                onehot_target = onehot_training(target, self.train_batch_size)
                #print(onehot_target.type())
                cur_loss = loss_function(forward_pass_output, onehot_target)
                cur_loss.backward()
                optimizer.step()
                loss += cur_loss
            # loss / number of batches
            avg_loss = loss / (len(self.train_loader.dataset) / self.train_batch_size)
            return avg_loss
        
        def testing():
            loss = 0
            correct = 0
            for batch_id, (data, target) in enumerate(self.test_loader):
                # data.view change the dimension of input to use forward function
                forward_pass_output = self.nn(data.view(self.test_batch_size, self.input_size))
                onehot_target = onehot_training(target, self.test_batch_size)
                cur_loss = loss_function(forward_pass_output, onehot_target)
                loss += cur_loss
                #print(forward_pass_output.size())
                #print(onehot_target.size())
                for i in range(self.test_batch_size):
                    val, position = torch.max(forward_pass_output.data[i], 0)
                 #   print('prediction = {}, actual = {}'.format(int(position), target[i]))
                    if position == target[i]:
                        correct += 1
            # loss / number of batches
            avg_loss = loss / (len(self.test_loader.dataset) / self.test_batch_size)
            accuracy = correct / len(self.test_loader.dataset)
            return avg_loss, accuracy

        for i in range(self.epoch):
            train_loss = training()
            test_loss,accuracy = testing()
            print('Epoch {}, training_loss = {}, testing_loss = {}, accuracy = {}'.format(i, train_loss, test_loss, accuracy))

        

    def forward(self, img):
        
        output = self.nn.forward(img.view(1, self.input_size))
        _, result = torch.max(output, 1)
        return result
