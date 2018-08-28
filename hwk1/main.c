#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

void c_conv(int in_channel, int o_channel, int kernel_size, int stride, double*** img, int img_height, int img_width);

void init_plot(double*** img, int height, int width, int in_channel);


int main(int argc, char** argv){
    int in_channel = 3;
    int kernel_size = 3;
    int stride = 1;

// Create Image matrix
    double *** img0 = (double ***) malloc(in_channel * sizeof(double**));
    double *** img1 = (double ***) malloc(in_channel * sizeof(double**));
    init_plot(img0, 720, 1280, 3);
    init_plot(img1, 1080, 1920, 3);
    char* filename = "partD_result.csv";

    FILE *fp;
    fp = fopen(filename, "w+");
    fprintf(fp, "Image, i, time\n");
    
    for (int i = 0; i < 12; i++){
        int o_c = (int)pow(2, i);
        clock_t s = clock();
        c_conv(in_channel, o_c, kernel_size, stride, img0, 720, 1280);
        clock_t e = clock();
        double elapse =  (double) (e - s) / CLOCKS_PER_SEC;
        fprintf(fp, "0,%d,%f\n", i, elapse);
        printf("img0: 0,%d,%f\n", i, elapse);
        
        // next for image 1
        s = clock();
        c_conv(in_channel, o_c, kernel_size, stride, img1, 1080, 1920);
        e = clock();
        elapse = (double) (e - s) / CLOCKS_PER_SEC;
        fprintf(fp, "1,%d,%f\n", i, elapse);
        printf("img1: 0,%d,%f\n", i, elapse);
    }
    fclose(fp); 
    return 0;
}

void c_conv(int in_channel, int o_channel, int kernel_size, int stride, double*** img, int img_height, int img_width){
    //printf("morning\n");
    double ***kernel = (double***) malloc(in_channel * sizeof(double**));
    for (int i = 0; i < in_channel; i++)
        kernel[i] = (double**) malloc(kernel_size * sizeof(double*));
    for (int i = 0; i < kernel_size; i++){
        for (int j = 0; j < kernel_size; j ++)
            kernel[i][j] = (double*) malloc(kernel_size* sizeof(double));
    }
    for (int i = 0; i < in_channel; i ++){
        for (int j = 0; j < kernel_size; j++){
            for (int k = 0; k < kernel_size; k++)
                kernel[i][j][k] = (double) (rand() % 10);
        }
    }
    int out_height = (img_height - kernel_size) /stride + 1;
    int out_width = (img_width - kernel_size) / stride + 1;
//    printf("in_channel = %d, o_channel = %d, kernel_size = %d,  stride = %d\n", in_channel, o_channel, kernel_size, stride);
    //printf("out_height = %d, out_width = %d\n", out_height, out_width);
    for (int ker = 0; ker < o_channel; ker ++){
        for (int i = 0; i < in_channel; i++){
            for (int j = 0; j < out_height; j++){
                for (int k = 0; k < out_width; k++){
                    double temp = 0;
                    for (int p = 0; p < in_channel; p++){
                        for (int m = 0; m < kernel_size; m++){
                            for (int n = 0; n < kernel_size; n++){
                                temp += kernel[p][m][n] * img[p][j + m][k + n];
                                //if (img_height == 1080 && j >= out_height - 360)
                                //printf("i = %d, j = %d, k = %d, p = %d, m = %d, n = %d\n", i, j, k, p, m, n);
                            }
                        }
                    }

                }
            }
        }
    }
};

void init_plot(double ***img, int height, int width, int in_channel){
    for (int i = 0; i < in_channel; i ++)
    {
        img[i] = (double**) malloc(height * sizeof(double*));
    }

    for (int i = 0; i < in_channel; i ++){
        for (int j = 0; j < height; j ++)
            img[i][j] = (double*) malloc(width * sizeof(double));
    }

    for (int i = 0; i < in_channel; i++){
        for (int j = 0; j < height; j ++){
            for (int k = 0; k < width; k++)
                img[i][j][k] = (i + k + j) % 255;
        }

    }

};

