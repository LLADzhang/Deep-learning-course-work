import numpy as np
import os
import shutil
from pprint import pprint as pp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from time import time, sleep
from torch.autograd import Variable

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)

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
    
class img2obj:

    def __init__(self):
        self.train_batch_size = 200 
        self.epoch = 50
        self.rate = 0.001 
        self.input_size = 32 * 32 * 3 #RGB 3 channels of data
        self.test_batch_size = 1000
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./cifar', 
                train=False, 
                download=True, 
                transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])),
                batch_size=self.test_batch_size, shuffle=True, num_workers=10)

        self.train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./cifar', 
                train=True, 
                download=True, 
                transform=transforms.Compose([transforms.ToTensor(), normalize])),
                batch_size=self.train_batch_size, shuffle=True, num_workers=10)

        self.classes = [
                'beaver', 'dolphin', 'otter', 'seal', 'whale', 
                'aquarium fish', 'flatfish', 'ray', 'shark', 'trout', 
                'orchids', 'poppies', 'roses', 'sunflowers', 'tulips', 
                'bottles', 'bowls', 'cans', 'cups', 'plates', 
                'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers', 
                'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
                'bed', 'chair', 'couch', 'table', 'wardrobe',
                'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', 
                'bear', 'leopard', 'lion', 'tiger', 'wolf', 
                'bridge', 'castle', 'house', 'road', 'skyscraper', 
                'cloud', 'forest', 'mountain', 'plain', 'sea',
                'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo', 
                'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
                'crab', 'lobster', 'snail', 'spider', 'worm', 
                'baby', 'boy', 'girl', 'man', 'woman', 
                'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
                'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
                'maple', 'oak', 'palm', 'pine', 'willow', 
                'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train', 
                'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'
                ]
        torch.manual_seed(1)
        self.labels = len(self.classes) 
        # input image is 3*32 * 32 so convert to 1D matrix
        self.model = LeNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.rate, weight_decay=0.0005)
        self.loss_function = nn.CrossEntropyLoss()
        self.check_point_file = 'img2obj_checkpoint.tar'

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
        _4d = torch.unsqueeze(img.type(torch.FloatTensor), 0)
        self.model.eval()
        output = self.model(_4d)
        _, result = torch.max(output, 1)
        return self.classes[result]

    def train(self, plot=False):
        print('training')

        def save(state, better, f=self.check_point_file):
            torch.save(state, f)
            if better:
                shutil.copyfile(f, 'img2obj_best.tar')

        def training():
            loss = 0
            self.model.train() # set to training mode
            for batch_id, (data, target) in enumerate(self.train_loader):
                # data.view change the dimension of input to use forward function
                forward_pass_output = self.model(data)
                
                cur_loss = self.loss_function(forward_pass_output, target)
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
                cur_loss = self.loss_function(forward_pass_output, target)
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
        '''
        for i in range(0, 10):
            label = self.classes[self.train_loader.dataset[i][1]]
            self.view(self.train_loader.dataset[i][0])
            print("actual label is", label)
            sleep(3)
        '''
        if plot == True:
            return self.time, self.training_loss, self.testing_loss, self.testing_acc

    def view(self, img):
        cate = self.forward(img)

        img = img.type(torch.FloatTensor) / 2 + 0.5
        img_numpy = np.transpose(img.numpy, (1,2,0))

        cv2.namedWindow(cate, cv2.WINDOW_NORMAL)
        cv2.imshow(cate, img_numpy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

