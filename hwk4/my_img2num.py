from pprint import pprint as pp
from neural_network import NeuralNetwork 
import torch
from torchvision import datasets, transforms
from time import time
import matplotlib.pyplot as plt

class MyImg2Num:
    def __init__(self):
        self.train_batch_size = 60
        self.epoch = 5
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
        self.nn = NeuralNetwork([self.input_size, 512, 256, 64, self.labels])

    def train(self):
        print('training')
        def onehot_training(target, batch_size):
                output = torch.zeros(batch_size, self.labels)
                for i in range(batch_size):
                    output[i][int(target[i])] = 1.0
                return output

        def training():
            loss = 0
            for batch_id, (data, target) in enumerate(self.train_loader):
                # data.view change the dimension of input to use forward function
                forward_pass_output = self.nn.forward(data.view(self.train_batch_size, self.input_size).type(torch.DoubleTensor))
                onehot_target = onehot_training(target, self.train_batch_size).type(torch.DoubleTensor)
                #print(onehot_target.type())
                self.nn.backward(onehot_target)
                loss += self.nn.total_loss
                self.nn.updateParams(self.rate)
            # loss / number of batches
            avg_loss = loss / (len(self.train_loader.dataset) / self.train_batch_size)
            return avg_loss
        
        def testing():
            loss = 0
            correct = 0
            for batch_id, (data, target) in enumerate(self.test_loader):
                # data.view change the dimension of input to use forward function
                forward_pass_output = self.nn.forward(data.view(self.test_batch_size, self.input_size).type(torch.DoubleTensor))
                onehot_target = onehot_training(target, self.test_batch_size).type(torch.DoubleTensor)
                loss += (onehot_target - forward_pass_output).pow(2).sum() / 2
                #print(forward_pass_output.size())
                #print(onehot_target.size())
                for i in range(self.test_batch_size):
                    val, position = torch.max(forward_pass_output[i], 0)
                 #   print('prediction = {}, actual = {}'.format(int(position), target[i]))
                    if position == target[i]:
                        correct += 1
            # loss / number of batches
            avg_loss = loss / len(self.test_loader.dataset) 
            accuracy = correct / len(self.test_loader.dataset)
            return avg_loss, accuracy
        acc_list = []
        train_loss_list = []
        test_loss_list = []
        speed = []

        for i in range(self.epoch):
            s = time()
            train_loss = training()
            e = time()
            test_loss,accuracy = testing()
            print('Epoch {}, training_loss = {}, testing_loss = {}, accuracy = {}, time = {}'.format(i, train_loss, test_loss, accuracy, e - s))
            acc_list.append(accuracy)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            speed.append(e-s)
        plt.plot(range(self.epoch), acc_list, 'r|--', label='Accuracy')
        plt.plot(range(self.epoch), train_loss_list, 'b*--', label='Training Loss')
        plt.plot(range(self.epoch), test_loss_list, 'yo--', label='Test Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.title('My Neural Network Evaluation')
        plt.savefig('my_compare.png')
        plt.clf()
        return speed
        

    def forward(self, img):
        
        output = self.nn.forward(img.view(1, self.input_size))
        _, result = torch.max(output, 1)
        return result
