#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "Read_bias.h"

using namespace cv;
using namespace std;

double getActivation(double input) 
{
    if(input < 0){
        input = 0;
    }
	return input;
}

void Free_memory(int num, int out_w, double*** output)
{
    for (int i = 0; i < num; i++){
		for (int j = 0; j < out_w; j++){
			free(*(*(output + i) + j));
		}
		free(*(output + i));
	}
	free(output);
}

void Convolution(int stride, int f_num, int f_dep, int out_w, int out_h, int f_w, int f_h, double*** input, double**** filter, double*** out_conv, double* conv_bias)
{
    for(int i=0; i < f_num; i++){      
        for(int j=0; j < f_dep; j++){
            for(int k=0 ; k < out_w ; k++){                       
                for(int l=0 ; l < out_h ; l++){               
                    for(int m=0; m < f_w; m++){               
                        for(int n=0; n < f_h; n++){
                            out_conv[i][k][l] += input[j][k*stride + m][l*stride + n] * filter[i][j][m][n];
                        }
                    }
                    out_conv[i][k][l] += conv_bias[i];
                }
            }
        }
    }
}

void Max_Pooling(int stride, int f_num, int out_pool_w, int out_pool_h, int pool_w, int pool_h, double*** out_conv, double*** out_pool)
{
    double max;

    for(int i=0; i < f_num; i++){      
        for(int j=0 ; j < out_pool_w ; j++){                       
            for(int k=0 ; k < out_pool_h ; k++){
                max = out_conv[i][j][k];               
                for(int l=0; l < pool_w; l++){               
                    for(int m=0; m < pool_h; m++){
                        if(max <  out_conv[i][j*stride + l][k*stride + m]){
                            max =  out_conv[i][j*stride + l][k*stride + m];
                        }
                    }
                }
                out_pool[i][j][k] = max;
            }
        }
    }
}

void Fully_Connected(double* fc_in, double* fc_out, double** fc_weight, int fc_num, int fc_w, double* ip_bias)
{
    int i, j;
    for(i=0; i < fc_num; i++){
        for(j=0; j < fc_w; j++){
            fc_out[i] += fc_in[j] * fc_weight[i][j];
        }
        fc_out[i] += ip_bias[i];
    }

    for(i=0; i < fc_num; i++){
        free(*(fc_weight + i));
    }
    free(fc_weight);
}

void Soft_max(double* fc2_out, int fc2_num)
{
    int i;
    double soft_max_sum, soft_max_max;

    soft_max_max = fc2_out[0];
    for(i=0; i < fc2_num; i++){
        if(soft_max_max < fc2_out[i]){
            soft_max_max = fc2_out[i];
        }
    }

    for(i=0; i < fc2_num; i++){
        soft_max_sum += exp(fc2_out[i] - soft_max_max);
    }

    for(i=0; i < fc2_num; i++){
        fc2_out[i] = exp(fc2_out[i] - soft_max_max)/soft_max_sum;
        printf("%d : %.4f  \n", i, fc2_out[i]);
    }
}

double*** Memory_alloc_3(int in_ch, int in_w, int in_h)
{
    double ***input;
	input = (double***)malloc(in_ch * sizeof(double**)); 
    for (int i = 0; i < in_ch; i++){
	    *(input + i) = (double**)malloc(in_w * sizeof(double*));
	    for (int j = 0; j < in_w; j++){
		    *(*(input + i) + j) = (double*)malloc(in_h * sizeof(double));
        }
    }
    return input;
}

int main()
{
    // variables
    int in_ch, in_w, in_h, stride, padding = 0;
    int f_num, f_dep, f_w, f_h;
    double ***input, ****filter, ***out_conv1, ***out_pool1, ***out_conv2, ***out_pool2;
    int x, y, z;
    int i, j, k, l, m, n, o;
    int fc1_num, fc1_w, fc2_num, fc2_w;
    double *fc1_in, **fc1_weight, *fc1_out, **fc2_weight, *fc2_out;

    ///////////////////////////////////////////////////////////////////////////////// Read Input Image
    Mat image;
    image = imread("/home/socmgr/iuy/convolution/numbers/test3.jpg",IMREAD_GRAYSCALE);  // Graysclae -> 1 channel
    if(image.empty())
    {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    in_ch = image.channels();
    in_w = image.rows;
    in_h = image.cols;

    // input image memory aloocation
    input = Memory_alloc_3(in_ch, in_w, in_h); 
    // input image memory initialization
    for (z = 0; z < in_ch; z++){       
		for (y = 0; y < in_w; y++){
			for (x = 0; x < in_h; x++){
				input[z][y][x] = image.at<uchar>(y, x);
                //printf("%3.f ",input[z][y][x]);
			}
            //printf("\n");
		}
        //printf("\n");
	}

    ///////////////////////////////////////////////////////////////////////////////// read 'Conv1' weight
    FILE *fp, *fp2;       // File pointer
    char dump[10];  // Dump string
    char cha;       // Dump char
    
    fp = fopen("/home/socmgr/iuy/convolution/weight_file/weight_lenet_origin.txt","r");
    if(fp == NULL){
        printf("Read Error\n");
        return 0;
    }

    fp2 = fopen("/home/socmgr/iuy/convolution/weight_file/bias.txt","r");
    if(fp2 == NULL){
        printf("Read Error\n");
        return 0;
    }
    
    fscanf(fp,"%s",dump);
    fscanf(fp,"%s",dump);
    fscanf(fp," [%d, %d, %d, %d],", &f_num, &f_dep, &f_w, &f_h);
    // 'conv1' output variables
    stride = 1;
    int out_w = ((in_w - f_w + 2*padding)/stride) + 1;
    int out_h = ((in_h - f_h + 2*padding)/stride) + 1;  
    // 4-dimension filter memory allocation
    filter = (double****)malloc(f_num * sizeof(double***));  
	for (i = 0; i < f_num; i++){
		*(filter + i) = (double***)malloc(f_dep * sizeof(double**));
		for (j = 0; j < f_dep; j++){
			*(*(filter + i) + j) = (double**)malloc(f_w * sizeof(double*));
            for(k=0; k < f_w; k++){
                *(*(*(filter + i) + j) + k) = (double*)malloc(f_h * sizeof(double));
            }
		}
    }
    // read weights
    fscanf(fp,"%s",dump); //data
	for (i = 0; i < f_num; i++){
		for (j = 0; j < f_dep; j++){
            for(k=0; k < f_w; k++){
                fscanf(fp," %c",&cha);  // [
                for(l=0; l < f_h; l++){
                    fscanf(fp,"%lf",&filter[i][j][k][l]);
                    fscanf(fp,"%c",&cha);   // ,
                    fscanf(fp,"%c",&cha);   // blank
                }
            }
        }
	}

    ///////////////////////////////////////////////////////////////////////////////// Read Bias
    double *conv1_bias, *conv2_bias, *ip1_bias, *ip2_bias;

    conv1_bias = Read_bias(fp2, conv1_bias, conv2_bias, ip1_bias, ip2_bias,0);
    conv2_bias = Read_bias(fp2, conv1_bias, conv2_bias, ip1_bias, ip2_bias,1);
    ip1_bias = Read_bias(fp2, conv1_bias, conv2_bias, ip1_bias, ip2_bias,2);
    ip2_bias = Read_bias(fp2, conv1_bias, conv2_bias, ip1_bias, ip2_bias,3);

    ///////////////////////////////////////////////////////////////////////////////// 'Conv1' convolution
    // 'conv1' output array memory allocation
    out_conv1 = Memory_alloc_3(f_num, out_w, out_h);
    //initialize 'conv1' array
	for (i = 0; i < f_num; i++){
		for (j = 0; j < out_w; j++){
		    for(k = 0; k < out_h; k++){
                out_conv1[i][j][k] = 0;
            }
        }
    }
    // 'Conv1' Convolution
    Convolution(stride, f_num, f_dep, out_w, out_h, f_w, f_h, input, filter, out_conv1, conv1_bias);
    // Free filter memory
    for (i = 0; i < f_num; i++){
	    for (j = 0; j < f_dep; j++){
            for(k = 0; k < f_w; k++){
                free(*(*(*(filter + i) + j) + k));
            }
            free(*(*(filter + i) + j));
	    }
	    free(*(filter + i));
    }
    free(filter);

    ///////////////////////////////////////////////////////////////////////////////// Max Pooling 1
    stride = 2;
    int pool1_w = 2, pool1_h = 2;
    int out_pool1_w = ((out_w - pool1_w + 2*padding)/stride) + 1;  
    int out_pool1_h = ((out_h - pool1_h + 2*padding)/stride) + 1; 
    double max;
    // 'Pool1' output array memory allocation
    out_pool1 = (double***)malloc(f_num * sizeof(double**));
	for (i = 0; i < f_num; i++){
		*(out_pool1 + i) = (double**)malloc(out_pool1_w * sizeof(double*));
		for (j = 0; j < out_pool1_w; j++){
		    *(*(out_pool1 + i) + j) = (double*)malloc(out_pool1_h * sizeof(double));
        }
    }
    // Max Pooling
    Max_Pooling(stride, f_num, out_pool1_w, out_pool1_h, pool1_w, pool1_h, out_conv1, out_pool1);

    ///////////////////////////////////////////////////////////////////////////////// read 'Conv2' weight
    int conv2_f_num, conv2_f_dep, conv2_f_w, conv2_f_h;

    fscanf(fp,"%s",dump);
    fscanf(fp,"%s",dump);
    fscanf(fp,"%s",dump);
    fscanf(fp," [%d, %d, %d, %d],", &conv2_f_num, &conv2_f_dep, &conv2_f_w, &conv2_f_h);
    // 'conv2' output variables
    stride = 1;
    int conv2_out_w = ((out_pool1_w - conv2_f_w + 2*padding)/stride) + 1; 
    int conv2_out_h = ((out_pool1_h - conv2_f_h + 2*padding)/stride) + 1; 
    // 4-dimension filter memory allocation
    filter = (double****)malloc(conv2_f_num * sizeof(double***)); 
	for (i = 0; i < conv2_f_num; i++){
		*(filter + i) = (double***)malloc(conv2_f_dep * sizeof(double**));
		for (j = 0; j < conv2_f_dep; j++){
			*(*(filter + i) + j) = (double**)malloc(conv2_f_w * sizeof(double*));
            for(k=0; k < conv2_f_w; k++){
                *(*(*(filter + i) + j) + k) = (double*)malloc(conv2_f_h * sizeof(double));
            }
		}
    }
    // read weights
    fscanf(fp,"%s",dump);
    fscanf(fp," %c",&cha);
	for (i = 0; i < conv2_f_num; i++){
		for (j = 0; j < conv2_f_dep; j++){
            for(k=0; k < conv2_f_w; k++){
                fscanf(fp," %c",&cha);
                for(l=0; l < conv2_f_h; l++){
                    fscanf(fp,"%lf",&filter[i][j][k][l]);
                    fscanf(fp,"%c",&cha);   
                    fscanf(fp,"%c",&cha);  
                }
            }
        }
	}

    ///////////////////////////////////////////////////////////////////////////////// 'Conv2' convolution
    // 'conv2' output array memory allocation
    out_conv2 = Memory_alloc_3(conv2_f_num, conv2_out_w, conv2_out_h);

    //initialize 'conv2' array
	for (i = 0; i < conv2_f_num; i++){
		for (j = 0; j < conv2_out_w; j++){
		    for(k = 0; k < conv2_out_h; k++){
                out_conv2[i][j][k] = 0;
            }
        }
    }
    // 'Conv2' convolution
    Convolution(stride, conv2_f_num, conv2_f_dep, conv2_out_w, conv2_out_h, conv2_f_w, conv2_f_h, out_pool1, filter, out_conv2, conv2_bias);
    // Free filter memory
    for (i = 0; i < conv2_f_num; i++){
	    for (j = 0; j < conv2_f_dep; j++){
            for(k = 0; k < conv2_f_w; k++){
                free(*(*(*(filter + i) + j) + k));
            }
            free(*(*(filter + i) + j));
	    }
	    free(*(filter + i));
    }
    free(filter);

    ///////////////////////////////////////////////////////////////////////////////// Max Pooling 2
    stride = 2;
    int pool2_w = 2, pool2_h = 2;
    int out_pool2_w = ((conv2_out_w - pool2_w + 2*padding)/stride) + 1; 
    int out_pool2_h = ((conv2_out_h - pool2_h + 2*padding)/stride) + 1; 
    // 'Pool1' output array memory allocation
    out_pool2 = Memory_alloc_3(conv2_f_num, out_pool2_w, out_pool2_h);

    // max pooling 2
    Max_Pooling(stride, conv2_f_num, out_pool2_w, out_pool2_h, pool2_w, pool2_h, out_conv2, out_pool2);

    ///////////////////////////////////////////////////////////////////////////////// read 'ip1'
    fc1_in = (double*)malloc(out_pool2_w * out_pool2_h * conv2_f_num * sizeof(double));
    // change array shape  50x4x4 -> 1x800
    for(i=0; i < conv2_f_num; i++){
        for(j=0; j < out_pool2_w; j++){
            for(k=0; k < out_pool2_h; k++){
                fc1_in[(out_pool2_w * out_pool2_h * i) + (j * out_pool2_h) + k] = out_pool2[i][j][k];
            }
        }
    }

    fscanf(fp,"%s",dump);
    fscanf(fp,"%s",dump);
    fscanf(fp,"%s",dump);
    fscanf(fp," [%d, %d],", &fc1_num, &fc1_w);
    // 'ip1' output array memory allocation
    fc1_out = (double*)malloc(fc1_num * sizeof(double));
    // 'ip1' weight array memory allocation
    fc1_weight = (double**)malloc(fc1_num * sizeof(double*)); 
	for (i = 0; i < fc1_num; i++){
		*(fc1_weight + i) = (double*)malloc(fc1_w * sizeof(double));
    }

    fscanf(fp,"%s",dump);
    for(i=0; i < fc1_num; i++){
        for(j=0; j < fc1_w; j++){
            fscanf(fp,"%c",&cha);
            fscanf(fp,"%c",&cha);
            fscanf(fp,"%lf",&fc1_weight[i][j]);
        }
        fscanf(fp,"%c",&cha);
        fscanf(fp,"%c",&cha);
    }

    ///////////////////////////////////////////////////////////////////////////////// Fully Connected 1
    Fully_Connected(fc1_in, fc1_out, fc1_weight, fc1_num, fc1_w, ip1_bias);

    ///////////////////////////////////////////////////////////////////////////////// ReLu Activation Fuction
    for(i=0; i < fc1_num; i++){
        fc1_out[i] = getActivation(fc1_out[i]);
    }

    ///////////////////////////////////////////////////////////////////////////////// Read 'ip2'
    fscanf(fp,"%s",dump);
    fscanf(fp,"%s",dump);
    fscanf(fp,"%s",dump);
    fscanf(fp," [%d, %d],", &fc2_num, &fc2_w);
    // 'ip2' output array memory allocation
    fc2_out = (double*)malloc(fc2_num * sizeof(double));
    // 'ip2' weight array memory allocation
    fc2_weight = (double**)malloc(fc2_num * sizeof(double*)); 
	for (i = 0; i < fc2_num; i++){
		*(fc2_weight + i) = (double*)malloc(fc2_w * sizeof(double));
    }

    fscanf(fp,"%s",dump);
    for(i=0; i < fc2_num; i++){
        for(j=0; j < fc2_w; j++){
            fscanf(fp,"%c",&cha);
            fscanf(fp,"%c",&cha);
            fscanf(fp,"%lf",&fc2_weight[i][j]);
        }
        fscanf(fp,"%c",&cha);
        fscanf(fp,"%c",&cha);
    }

    ///////////////////////////////////////////////////////////////////////////////// Fully Connected 2
    Fully_Connected(fc1_out, fc2_out, fc2_weight, fc2_num, fc2_w, ip2_bias);

	for (i = 0; i < fc2_num; i++){
                printf("%5f  ", fc2_out[i]);
    }
    printf("\n");

    ///////////////////////////////////////////////////////////////////////////////// Soft Max
    Soft_max(fc2_out, fc2_num);

    ///////////////////////////////////////////////////////////////////////////////// Free Memory 
    free(fc1_in);
    free(fc1_out);
    free(fc2_out);

    free(conv1_bias);
    free(conv2_bias);
    free(ip1_bias);
    free(ip2_bias);

    Free_memory(conv2_f_num, out_pool2_w, out_pool2);
    Free_memory(conv2_f_num, conv2_out_w, out_conv2);
    Free_memory(f_num, out_pool1_w, out_pool1);
    Free_memory(f_num, out_w, out_conv1);
    Free_memory(in_ch, in_w, input);
}
