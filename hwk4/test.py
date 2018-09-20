from my_img2num import MyImg2Num
from nn_img2num import NNImg2Num
print('running self nn')
my_img_2num = MyImg2Num()
my_img_2num.train()
print('running library nn')
nn_img = NNImg2Num()
nn_img.train()
