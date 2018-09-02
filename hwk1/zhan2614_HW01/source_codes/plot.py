import matplotlib.pyplot as plt
from sys import argv
import csv


with open(argv[1], 'r') as csv_in:
    reader = list(csv.reader(csv_in))
    img_0_x = []
    img_1_x = []
    img_0_y = []
    img_1_y = []
    xlabel = reader[0][1].capitalize()
    ylabel = reader[0][2].capitalize()
    for row in reader[1:]:
        if row[0] == '0':
            img_0_x.append(row[1])
            img_0_y.append(row[2])
        else:
            img_1_x.append(row[1])
            img_1_y.append(row[2])
    # result figure for image 0
    plt.plot(img_0_x, img_0_y, '--bo')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs " + xlabel + " of image 0 (1280 x 720)")
    plt.grid()
    plt.savefig(argv[1].split('_')[0] + '_image0.png')
    plt.clf()
    # result figure for image 1
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(img_1_x, img_1_y, '--bo')
    plt.xlabel = xlabel
    plt.ylabel = ylabel
    plt.title(ylabel + " vs " + xlabel + " of image 1 (1920 x 1080)")
    plt.grid()
    plt.savefig(argv[1].split('_')[0] + '_image1.png')

