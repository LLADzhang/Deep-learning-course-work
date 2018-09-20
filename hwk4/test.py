from my_img2num import MyImg2Num

import matplotlib.pyplot as plt
from nn_img2num import NNImg2Num
print('running self nn')
my_img_2num = MyImg2Num()
my_time = my_img_2num.train()
print(my_time)
print('running library nn')
nn_img = NNImg2Num()
nn_time = nn_img.train()
print(nn_time)
data = [my_time, nn_time]
plt.boxplot(data)
plt.xticks(range(2), ['MyImg2Num', 'NNImg2Num'])
plt.ylabel('Running Time in Seconds')
plt.savefig('efficiency.png')
