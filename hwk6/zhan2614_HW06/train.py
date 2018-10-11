import argparse
import matplotlib.pyplot as plt
import os
import shutil
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from time import time


class AlexNet(nn.Module):
    def __init__(self, num_classes=200):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # the fully connected layer
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Model:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--data', type=str, help='Directory to thee tiny image set')
        parser.add_argument('--save', type=str, help='Directory to save trained model after completion of training')
        args = parser.parse_args()


        self.train_batch_size =  100
        self.epoch = 51
        self.rate = 0.1 
        self.val_batch_size = 10
        def create_val_folder():
            path = os.path.join(args.data, 'val/images')  # path where validation data is present now
            filename = os.path.join(args.data, 'val/val_annotations.txt')  # file where image2class mapping is present
            fp = open(filename, "r")  # open file in read mode
            data = fp.readlines()  # read line by line

            # Create a dictionary with image names as key and corresponding classes as values
            val_img_dict = {}
            for line in data:
                words = line.split("\t")
                val_img_dict[words[0]] = words[1]
            fp.close()

            # Create folder if not present, and move image into proper folder
            for img, folder in val_img_dict.items():
                newpath = (os.path.join(path, folder))
                if not os.path.exists(newpath):  # check if folder exists
                    os.makedirs(newpath)

                if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
                    os.rename(os.path.join(path, img), os.path.join(newpath, img))
        create_val_folder()
	
        # load data: from https://github.com/pytorch/examples/blob/master/imagenet/main.py

        traindir = os.path.join(args.data, 'train')
        if not os.path.exists(traindir):
            os.makedirs(traindir)
        valdir = os.path.join(args.data, 'val/images')

        if not os.path.exists(valdir):
            os.makedirs(valdir)
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
	    traindir,
	    transforms.Compose([
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize,
        ]))
        
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = self.train_batch_size, shuffle=True, num_workers = 5)        
        
        val_dataset = datasets.ImageFolder(
	    valdir,
            transforms.Compose([
                    transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize, 
        ]))
        
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size = self.val_batch_size, shuffle=True, num_workers = 5)
        def get_tiny_classes(class_list):
            fp = open(os.path.join(args.data, 'words.txt'))
            whole_class_dict = {}
            for line in fp.readlines():
                fields = line.split("\t")
                super_label = fields[1].split(',')
                whole_class_dict[fields[0]] = super_label[0].rstrip()
            fp.close()

            tiny_class = {}
            for lab in class_list:
                for k,v in whole_class_dict.items():
                    if lab == k:
                        tiny_class[k] = v
                        continue
            return tiny_class
       
        self.classes = train_dataset.classes
        self.tiny_classes = get_tiny_classes(self.classes)
	    
        pretrained_model = models.alexnet(pretrained=True)
        torch.manual_seed(1)
        self.model = AlexNet()
        # To copy parameters
        for i, j in zip(self.model.modules(), pretrained_model.    modules()):  # iterate over both models
             if not list(i.children()):
                 if len(i.state_dict()) > 0:  # copy weights only     for the convolution and linear layers
                     if i.weight.size() == j.weight.size():  # this helps to prevent copying of weights of last layer
                         i.weight.data = j.weight.data
                         i.bias.data = j.bias.data
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.classifier[6].parameters():
            p.requires_grad = True


        self.optimizer = torch.optim.Adam(self.model.classifier[6].parameters(), lr=self.rate)
        self.loss_function = nn.CrossEntropyLoss()

        self.check_point_file = os.path.join(args.save, 'alex_checkpoint.tar')
        if not os.path.exists(os.path.dirname(self.check_point_file)):
            try:
                os.makedirs(os.path.dirname(self.check_point_file))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        if os.path.isfile(self.check_point_file):
            cp = torch.load(self.check_point_file)
            self.start = cp['epoch']
            self.best_acc = cp['best_acc']

            print('checkpoint found at epoch', self.start)
            self.model.load_state_dict(cp['model'])
            self.optimizer.load_state_dict(cp['optimizer'])

            self.training_loss = cp['training_loss']
            self.training_acc = cp['training_acc']
            self.testing_loss = cp['testing_loss']
            self.testing_acc = cp['testing_acc']
            self.time = cp['time']
        else:
            self.start = 0
            self.best_acc = 0 

            self.training_loss = []
            self.training_acc = []
            self.testing_loss = []
            self.testing_acc = []
            self.time = []
    def train(self, plot=False):
        def save(state, better, f=self.check_point_file):
            torch.save(state, f)
            if better:
                shutil.copyfile(f, os.path.join(args.save, 'alexnet_best.tar'))
        def training():
            correct = 0
            loss = 0
            self.model.train() # set to training mode
            for batch_id, (data, target) in enumerate(self.train_loader):
                # data.view change the dimension of input to use forward function
                forward_pass_output = self.model(data)
                #print(onehot_target.type())
                cur_loss = self.loss_function(forward_pass_output, target)
                loss += cur_loss.data
                self.optimizer.zero_grad()
                cur_loss.backward()
                self.optimizer.step()
                
                val, position = torch.max(forward_pass_output.data, 1)
                for i in range(self.train_batch_size):
                    if position[i] == target.data[i]:
                        correct += 1
            # loss / number of batches
            avg_loss = loss / (len(self.train_loader.dataset) / self.train_batch_size)
            accuracy = correct / len(self.train_loader.dataset)
            
            return avg_loss, accuracy
        
        def testing():
            self.model.eval()
            loss = 0
            correct = 0
            for batch_id, (data, target) in enumerate(self.val_loader):
                # data.view change the dimension of input to use forward function
                forward_pass_output = self.model(data)
                cur_loss = self.loss_function(forward_pass_output, target)
                loss += cur_loss.data
                #print(forward_pass_output.size())
                #print(onehot_target.size())
                val, position = torch.max(forward_pass_output.data, 1)
                for i in range(self.val_batch_size):
                 #   print('prediction = {}, actual = {}'.format(int(position), target[i]))
                    if position[i] == target[i]:
                        correct += 1
            # loss / number of batches
            avg_loss = loss / (len(self.val_loader.dataset) / self.val_batch_size)
            accuracy = correct / len(self.val_loader.dataset)
            return avg_loss, accuracy

        for i in range(self.start + 1, self.epoch + 1):
            print('Epoch {}'.format(i))
            s = time()
            print('Training\n')
            train_loss, train_accuracy = training()
            e = time()
            print('Training Done. Testing....\n')
            test_loss, test_accuracy = testing()
            self.testing_acc.append(test_accuracy)
            self.training_acc.append(train_accuracy)
            self.training_loss.append(train_loss)
            self.testing_loss.append(test_loss)
            self.time.append(e-s)
            better = False
            if test_accuracy > self.best_acc:
                better = True
            self.best_acc = max(self.best_acc, test_accuracy)
            print('training_loss = {}, testing_loss = {}, training accuracy = {}, testing accuracy = {}, current best test accuracy = {}, time = {}, better = {}'.format(train_loss, test_loss, train_accuracy, test_accuracy, self.best_acc, e - s, better))
            print('Saved checkpoint at', i)
            state = {
                    'epoch': i,
                    'best_acc': self.best_acc,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'training_loss': self.training_loss,
                    'testing_loss': self.testing_loss,
                    'testing_acc': self.testing_acc,
                    'training_acc': self.training_acc,
                    'time': self.time,
                    'classes': self.classes,
                    'tiny_class': self.tiny_classes

                    }
            save(state,better) 
        if plot == True:
            return self.time, self.training_loss, self.testing_loss, self.training_acc, self.testing_acc


def graph(time, train_loss, test_loss, train_accuracy, test_accuracy, name):

    plt.plot(time, 'k*:')
    plt.ylabel('Running Time in Seconds')
    plt.legend()
    plt.xlabel('Epoch')
    plt.title("Running time of " + name)
    plt.savefig(name+'_time.png')
    plt.clf()

    plt.plot(range(len(train_accuracy)), train_accuracy, 'r*--', label='Training Accuracy')
    plt.plot(range(len(test_accuracy)), test_accuracy, 'b.--', label='Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title("Accuracy of " + name)
    plt.legend()
    plt.savefig(name + '_acc.png')
    plt.clf()
    plt.plot(range(len(train_loss)), train_loss, 'r*--', label='Training loss')
    plt.plot(range(len(test_loss)), test_loss, 'bo-.', label='Testing loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title("Loss of "+name)
    plt.ylabel('Loss')
    plt.savefig(name+'_loss.png')
    plt.clf()


if __name__ == "__main__":
    m = Model()
    time, train_loss, test_loss, train_accuracy, test_accuracy = m.train(True)
    graph(time, train_loss, test_loss, train_accuracy, test_accuracy,  'Tiny Image')
