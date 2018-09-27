import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from time import time
from torch.autograd import Variable


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2) 
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class img2num:

    def __init__(self):
        self.train_batch_size = 60
        self.epoch = 50
        self.labels = 10
        self.rate = 1 
        self.input_size = 28 * 28
        self.test_batch_size = 1000
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./mnist', 
                train=False, 
                download=True, 
                transform=transforms.Compose([transforms.ToTensor()])),
                batch_size=self.test_batch_size, shuffle=True, num_workers=10)

        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./mnist', 
                train=True, 
                download=True, 
                transform=transforms.Compose([transforms.ToTensor()])),
                batch_size=self.train_batch_size, shuffle=True, num_workers=10)
        
        # input image is 28 * 28 so convert to 1D matrix
        # output labels are 10 [0 - 9]
        torch.manual_seed(1)
        self.model = LeNet()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.rate)
        self.loss_function = nn.MSELoss()

        self.check_point_file = 'img2num_checkpoint.tar'

        if os.path.isfile(self.check_point_file):
            cp = torch.load(self.check_point_file)
            self.start = cp['epoch']
            self.best_acc = cp['best_acc']

            print('checkpoint found at epoch', self.start)
            self.model.load_state_dict(cp['model'])
            self.optimizer.load_state_dict(cp['optimizer'])

            self.training_loss = cp['training_loss']
            self.testing_loss = cp['testing_loss']
            self.testing_acc = cp['testing_acc']
            self.time = cp['time']
        else:
            self.start = 0
            self.best_acc = 0 

            self.training_loss = []
            self.testing_loss = []
            self.testing_acc = []
            self.time = []

  
        # img is 28*28 bytetensor 
    def forward(self, img):
        _3d = torch.unsqueeze(img, 0)
        _4d = torch.unsqueeze(_3d, 0)
        self.model.eval()
        output = self.model(_4d)
        _, result = torch.max(output, 1)
        return result

    def train(self, plot=False):
        print('training')
        def save(state, better, f=self.check_point_file):
            torch.save(state, f)
            if better:
                shutil.copyfile(f, 'img2num_best.tar')

        def onehot_training(target, batch_size):
                output = torch.zeros(batch_size, self.labels)
                for i in range(batch_size):
                    output[i][int(target[i])] = 1.0
                return output

        def training():
            loss = 0
            self.model.train() # set to training mode
            for batch_id, (data, target) in enumerate(self.train_loader):
                # data.view change the dimension of input to use forward function
                forward_pass_output = self.model(data)
                onehot_target = onehot_training(target, self.train_batch_size)
                #print(onehot_target.type())
                cur_loss = self.loss_function(forward_pass_output, onehot_target)
                loss += cur_loss.data
                self.optimizer.zero_grad()
                cur_loss.backward()
                self.optimizer.step()
            # loss / number of batches
            avg_loss = loss / (len(self.train_loader.dataset) / self.train_batch_size)
            return avg_loss
        
        def testing():
            self.model.eval()
            loss = 0
            correct = 0
            for batch_id, (data, target) in enumerate(self.test_loader):
                # data.view change the dimension of input to use forward function
                forward_pass_output = self.model(data)
                onehot_target = onehot_training(target, self.test_batch_size)
                cur_loss = self.loss_function(forward_pass_output, onehot_target)
                loss += cur_loss.data
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
        for i in range(self.start + 1, self.epoch + 1):
            s = time()
            train_loss = training()
            e = time()
            test_loss,accuracy = testing()
            print('Epoch {}, training_loss = {}, testing_loss = {}, accuracy = {}, time = {}'.format(i, train_loss, test_loss, accuracy, e - s))
            self.testing_acc.append(accuracy)
            self.training_loss.append(train_loss)
            self.testing_loss.append(test_loss)
            self.time.append(e-s)
            better = False
            if accuracy > self.best_acc:
                better = True
            self.best_acc = max(self.best_acc, accuracy)
            print('Save checkpoint at', i)
            state = {
                    'epoch': i,
                    'best_acc': self.best_acc,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'training_loss': self.training_loss,
                    'testing_loss': self.testing_loss,
                    'testing_acc': self.testing_acc,
                    'time': self.time
                    }
            save(state,better) 
     
        if plot == True:
            return speed, train_loss_list, test_loss_list, acc_list

'''
        plt.plot(range(self.epoch), acc_list, 'r|--', label='Accuracy')
        plt.plot(range(self.epoch), train_loss_list, 'b*--', label='Training Loss')
        plt.plot(range(self.epoch), test_loss_list, 'yo--', label='Test Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.title('Library Neural Network Evaluation')
        plt.savefig('nn_compare.png')
        plt.clf()
    '''
    
