import matplotlib.pyplot as plt
from img2num import img2num
from img2obj import img2obj
print('img2Num testing')
img = img2num()
time, train_loss, test_loss, accuracy = img.train(True)
print(time, train_loss, test_loss, accuracy)
print('img2obj testing')
img = img2obj()
time, train_loss,test_loss, accuracy = img.train(True)
print(time, train_loss, test_loss, accuracy)
img.cam(0)
'''
data = [my_time, nn_time]
plt.boxplot(data)
plt.xticks(range(1, 3), ['MyImg2Num', 'NNImg2Num'])
plt.ylabel('Running Time in Seconds')
plt.legend()
plt.savefig('efficiency.png')
plt.clf()

plt.plot(range(30), nn_accuracy, 'r*--', label='Accuracy of torch nn')
plt.plot(range(30), my_accuracy, 'b|--', label='Accuracy of my nn')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy.png')
plt.clf()

plt.plot(range(30), my_train_loss, 'r*--', label='Training loss of my nn')
plt.plot(range(30), my_test_loss, 'ro-.', label='Testing loss of my nn')
plt.plot(range(30), nn_train_loss, 'b1--', label='Training loss of torch nn')
plt.plot(range(30), nn_test_loss, 'b+-.', label='Testing loss of torch nn')
plt.xlabel('Epoch')
plt.legend()
plt.ylabel('Loss')
plt.savefig('loss.png')
plt.clf()
'''
