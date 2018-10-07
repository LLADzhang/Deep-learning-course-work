import argparse
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision import datasets, transforms

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='path to the tiny image set')
parser.add_argument('--save', type=str, help='path to directory to save trained model after completion of training')
# parse_args returns an object with attributes as defined in the add_argument. The ArgumentParser parses command line
# arguments from sys.argv, converts to appropriate type and takes defined action (default: 'store')
args = parser.parse_args()



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
        x = F.softmax(x)
        return x


def alexnet(pretrained=False, **kwargs):
    """
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model


class Model:
    def __init__(self):
        self.train_batch_size = 60
        self.epoch = 50
        self.labels = 10
        self.rate = 1 
        self.input_size = 28 * 28
        self.val_batch_size = 1000
	# load data: from https://github.com/pytorch/examples/blob/master/imagenet/main.py
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
	    traindir,
	    transforms.Compose([
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize,
        ]))
        
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size = self.train_batch_size, shuffle=True, num_workers = 5)        
        val_dataset = datasets.ImageFolder(
	    valdir,
            transforms.Compose([
                    transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize, 
        ]))
        def get_tiny_classes(class_list):
            fp = open(os.path.join(args.data, 'classes.txt'))
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
        self.tiny_classes = get_classes(self.classes)
	    
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size = self.val_batch_size, shuffle=True, num_workers = 5)
        pretrained_model = alexnet(True)
        # input image is 28 * 28 so convert to 1D matrix
        # output labels are 10 [0 - 9]
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
        for p in self.model.classfier[6].parameters():
            p.requires_grad = True


        self.optimizer = torch.optim.Adam(self.model.classfier[6].parameters(), lr=self.rate)
        self.loss_function = nn.CrossEntropyLoss()

        self.check_point_file = 'alex_checkpoint.tar'

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
    def train(self):
        def save(state, better, f=self.check_point_file):
            torch.save(state, f)
            if better:
                shutil.copyfile(f, 'img2num_best.tar')
        def training():
            correct = 0
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
            for batch_id, (data, target) in enumerate(self.test_loader):
                # data.view change the dimension of input to use forward function
                forward_pass_output = self.model(data)
                cur_loss = self.loss_function(forward_pass_output, target)
                loss += cur_loss.data
                #print(forward_pass_output.size())
                #print(onehot_target.size())
                val, position = torch.max(forward_pass_output.data, 1)
                for i in range(self.test_batch_size):
                 #   print('prediction = {}, actual = {}'.format(int(position), target[i]))
                    if position == target[i]:
                        correct += 1
            # loss / number of batches
            avg_loss = loss / (len(self.test_loader.dataset) / self.test_batch_size)
            accuracy = correct / len(self.test_loader.dataset)
            return avg_loss, accuracy

        for i in range(self.start + 1, self.epoch + 1):
            s = time()
            train_loss, train_accuracy = training()
            e = time()
            test_loss, test_accuracy = testing()
            print('Epoch {}, training_loss = {}, testing_loss = {}, accuracy = {}, time = {}'.format(i, train_loss, test_loss, accuracy, e - s))
            self.testing_acc.append(test_accuracy)
            self.training_acc.append(train_accuracy)
            self.training_loss.append(train_loss)
            self.testing_loss.append(test_loss)
            self.time.append(e-s)
            better = False
            if test_accuracy > self.best_acc:
                better = True
            self.best_acc = max(self.best_acc, test_accuracy)
            print('Save checkpoint at', i)
            state = {
                    'epoch': i,
                    'best_acc': self.best_acc,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'training_loss': self.training_loss,
                    'testing_loss': self.testing_loss,
                    'testing_acc': self.testing_acc,
                    'training_acc': self.training_acc,
                    'time': self.time
                    }
            save(state,better) 
     
        if plot == True:
            return self.time, self.training_loss, self.testing_loss, self.testing_acc

if __name__ == "__main__":
    m = Model()
    m.train()
