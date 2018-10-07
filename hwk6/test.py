import matplotlib.pyplot as plt
from img2num import img2num
from img2obj import img2obj
def graph(time, train_loss, test_loss, accuracy, name):

    plt.plot(time, 'k*:')
    plt.ylabel('Running Time in Seconds')
    plt.legend()
    plt.xlabel('Epoch')
    plt.title("Running time of " + name)
    plt.savefig(name+'_time.png')
    plt.clf()

    plt.plot(range(len(accuracy)), accuracy, 'r*--', label='Testing Accuracy')
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
print('img2Num testing')
img = img2num()
time, train_loss, test_loss, accuracy = img.train(True)
graph(time, train_loss, test_loss, accuracy, 'img2num')
print('img2obj testing')
img = img2obj()
time, train_loss,test_loss, accuracy = img.train(True)
graph(time, train_loss, test_loss, accuracy, 'img2obj')
img.cam(0)
