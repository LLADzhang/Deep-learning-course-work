from conv import Conv2D
from PIL import Image
from torchvision.transforms import Compose, ToTensor
from torchvision.utils import save_image
import torch
from sys import argv
from time import time
import csv

transform = Compose([ToTensor()])
# transform a image to tensor (channel, height, width)

img1 = Image.open("img1.jpg")
img0 = Image.open('img0.jpg')
img_tensors = [transform(img0), transform(img1)]
if len(argv) != 2:
    print("Usage: python main.py part[A, B, C]")
    exit(1)

tasks = [[3,1,3,1], [3,2,5,1], [3,3,3,2]]
if argv[1] == 'partA':
    # Part A
    for tsk_id in range(len(tasks)):
        task = tasks[tsk_id]
        print('Part A Task ' + str(tsk_id + 1))

        for img_id in range(len(img_tensors)):

            print("Image ", img_id, "size: ", img_tensors[img_id].size())
            conv = Conv2D(task[0],task[1],task[2],task[3],)
            ops, output_img = conv.forward(img_tensors[img_id])
            print('Total operation', ops, ', output tensor size:', output_img.size())
            num_channels = output_img.size()[0]
            if num_channels == 1:
                # task 1
                file_name = "image" + str(img_id) + "/plt_" + str(img_id) + "_partA_task" + str(tsk_id + 1) + "_k1.jpg"       
                print("Save to", file_name, "\n")
                save_image(output_img, file_name)

            elif num_channels == 2:
                # task 2
                for i in range(num_channels):
                    file_name = "image" + str(img_id) + "/plt_" + str(img_id) + "_partA_task" + str(tsk_id + 1) + "_k" + str(i + 4) +".jpg"       
                    print("Save to", file_name, "\n")
                    save_image(output_img[i, :, :], file_name)
            else:
                # task 3 with 3 o channels
                for i in range(num_channels):
                    file_name = "image" + str(img_id) + "/plt_" + str(img_id) + "_partA_task" + str(tsk_id + 1) + "_k" + str(i + 1) +".jpg"       
                    print("Save to", file_name, "\n")
                    save_image(output_img[i, :, :], file_name)
elif argv[1] == 'partB':
    print("Part B")
    task = tasks[0]

    with open('partB_result.csv', 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['image', 'i', 'computing time'])
        for img_id in range(len(img_tensors)):
            print("Image " + str(img_id))
            print("Image, i, Time")
            for i in range(11):
                o_c = 2**i
                s = time()
                conv = Conv2D(task[0], o_c, task[2], task[3], 'rand')
                ops, output_img = conv.forward(img_tensors[img_id])
                e = time()
                row = (img_id, i, e-s)
                print(row)
                csv_out.writerow(row)

elif argv[1] == 'partC':
    print("part C")
    task = tasks[1]

    with open('partC_result.csv', 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['image', 'kernel size', 'operations'])
        for img_id in range(len(img_tensors)):
            print("Image " + str(img_id))
            print("Image, Kernel Size, Operations")
            for ker_size in range(3, 12, 2):
                conv = Conv2D(task[0], task[1], ker_size, task[3], 'rand')
                ops, output_img = conv.forward(img_tensors[img_id])
                row = (img_id, ker_size, ops)
                print(row)
                csv_out.writerow(row)
        
else:
    print("Wrong argument", argv[1])
    print("Abort!")

