import torch


class Conv2D:
    def __init__(self, in_channel, o_channel, kernel_size, stride, mode='known'):
        self.in_channel = in_channel
        self.o_channel = o_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode
        # predefined kernels
        self.k1 = torch.FloatTensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        self.k2 = torch.FloatTensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

        self.k3 = torch.FloatTensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        self.k4 = torch.FloatTensor([[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        self.k5 = torch.FloatTensor([[-1, -1, 0, 1, 1], [-1, -1, 0, 1, 1],[-1, -1, 0, 1, 1],[-1, -1, 0, 1, 1],[-1, -1, 0, 1, 1]])

    def forward(self, img):
        # img is a 3D FloatTensor
        # output is a tuple of (# of operations, 3D FloatTensor impage)
        # the 3D FloatTensor image is a greyscale image with 0s on the third domain.

        channels, height, width = img.size()
        #print("img height:", height, "img width", width, "img stride", self.stride)
        out_height = int((( height - self.kernel_size ) / self.stride + 1 ))
        out_width = int((( width - self.kernel_size ) / self.stride + 1 ))
        out_img = torch.zeros(self.o_channel, out_height, out_width)
        #print("out_height:", out_height, "out_width", out_width)
        mul_cnt = 0
        add_cnt = 0
        if self.mode == 'known':
            # task 1
            if self.o_channel == 1:
                kernel = torch.stack([self.k1 for i in range(self.in_channel)])
                kernels = [kernel]
            elif self.o_channel == 2:
                # task 2
                kernel1 = torch.stack([self.k4 for i in range(self.in_channel)])
                kernel2 = torch.stack([self.k5 for i in range(self.in_channel)])
                kernels = [kernel1, kernel2]
           
            elif self.o_channel == 3:
                # task 3
                kernel1 = torch.stack([self.k1 for i in range(self.in_channel)])
                kernel2 = torch.stack([self.k2 for i in range(self.in_channel)])
                kernel3 = torch.stack([self.k3 for i in range(self.in_channel)])
                kernels = [kernel1, kernel2, kernel3]

            for ind in range(self.o_channel):
                for row in range(out_height):
                    for col in range(out_width):
                        temp_tensor = torch.mul(kernels[ind], img[:, row * self.stride : row * self.stride + self.kernel_size, col * self.stride : col * self.stride + self.kernel_size])
                        out_img[ind, row, col] = temp_tensor.sum()
                        mul_cnt += 1
                        add_cnt += 1
                # change the output image to 3D float tensor
                ops = mul_cnt * self.kernel_size **2 + add_cnt * ( self.kernel_size **2 - 1)
            return ops, out_img
        elif self.mode == 'rand':
            # for part B 
            kernels = []
            for i in range(self.o_channel):
                k = torch.stack([torch.rand(self.kernel_size, self.kernel_size) for i in range(self.in_channel)])
                kernels.append(k)

            for ind in range(self.o_channel):
                for row in range(out_height):
                    for col in range(out_width):
                        temp_tensor = torch.mul(kernels[ind], img[:, row : row + self.kernel_size, col : col + self.kernel_size])
                        out_img[ind, row, col] = temp_tensor.sum()
                        mul_cnt += 1
                        add_cnt += 1
                # change the output image to 3D float tensor
                ops = mul_cnt * self.kernel_size **2 + add_cnt * ( self.kernel_size **2 - 1)
            return ops, out_img

        else:
            print("unknown mode: " + self.mode)
            exit(1)
