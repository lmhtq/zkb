#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <algorithm>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUBLAS_ERROR_CHECK(sdata) if(CUBLAS_STATUS_SUCCESS!=sdata){printf("ERROR at:%s:%d\n",__FILE__,__LINE__);exit(-1);}


using namespace cv;
using namespace std;


#define nbins 30
#define width 2
#define WIDTH 5

#define T 12.0f
#define PI 3.141593
//feature
float feature[nbins];
float gaussker[WIDTH*WIDTH] = 
{
1,4,7,4,1,
4,16,26,16,4,
7,26,41,26,7,
4,16,26,16,4,
1,4,7,4,1,
 };//主函数开始就 除以 273

extern "C"

//
__global__ void
calcEOH(float *d_fgray, uchar *gray, float *theta, float *d_gaussker, int row, int col)
{
    __shared__ float shgaussker[25];
    if (threadIdx.x < 25)
        shgaussker[threadIdx.x] = d_gaussker[threadIdx.x];
    __syncthreads();
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
    int len = row * col;
    while (i < len)
    {
        d_fgray[i] = (float)gray[i];
        i += stride;
    }
    __syncthreads();
    
    //guass filter
    i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < len)
    {
        float tmp = 0.0f;
        int r = i / col;
        int c = i % col;
        
        for (int ii = -width; ii <= width; ii++)
        {
            for (int jj = -width; jj <= width; jj++)
            {
                int ir = r + ii, ic = c +jj;
                if (ir < 0) ir = -ir;
                if (ir >= row) ir = 2*(row-1) - ir;
                if (ic < 0) ic = -ic;
                if (ic >= col) ic = 2*(col-1) - ic;
                tmp += d_fgray[ir * col + ic] * shgaussker[ (ii+width)*WIDTH + (jj+width) ];
            }
        }
        
        d_fgray[i] = tmp;
        
        i += stride;
    }
    
    __syncthreads();
    //canny filter 
    int cannyx[4] = {-1,1,-1,1};
    int cannyy[4] = {1,1,-1,-1};
    i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < len)
    {
        int r = i / col;
        int c = i % col;
        float tmpx = 0.0f;
        float tmpy = 0.0f;
        
        for (int ii = 0; ii <= 1; ii++)
        {
            for (int jj = 0; jj <= 1; jj++)
            {
                int ir = r + ii, ic = c +jj;
                if (ir < 0) ir = -ir;
                if (ir >= row) ir = 2*(row-1) - ir;
                if (ic < 0) ic = -ic;
                if (ic >= col) ic = 2*(col-1) - ic;
                tmpx += d_fgray[ir * col + ic] * cannyx[ ii * 2 + jj ];
                tmpy += d_fgray[ir * col + ic] * cannyy[ ii * 2 + jj ];
                
            }
        }
        
        if(sqrtf( tmpx*tmpx + tmpy*tmpy ) > T)
            theta[i] = atan( tmpy/tmpx ) / PI * 180.0f + 180.0f;
        
        i += stride;
    }
    __syncthreads();   
}

//hist
__global__ void
calcHist(float *d_hist, int *d_hist_int, float *theta, int row, int col )
{
    __shared__ int tmp[nbins];
    if (threadIdx.x < nbins)
    {
        tmp[threadIdx.x] = 0;
    }
    __syncthreads();
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
    int len = row * col;
    while (i < len)
    {
        int ind = (int)theta[i] / (180 / nbins);
        ind %= nbins;
        atomicAdd( &tmp[ind], 1 );
        d_hist_int[i] = 0;
        i += stride;
    }
    __syncthreads();
    
    if (threadIdx.x < nbins)
        atomicAdd( &d_hist_int[threadIdx.x], tmp[threadIdx.x] );
    __syncthreads();
    
    i = blockDim.x * blockIdx.x + threadIdx.x;
	while (i < nbins)
    {
        d_hist[i] = (float)d_hist_int[i] / (float)(len);
        i += stride;
    }
}

void print_feature()
{
    for (int i = 0; i < nbins; i++) 
    {
        printf("%f ", feature[i]);
    }
    printf("\n");
}

void check(int argc, char** argv, char *name)
{
    if (argc < 2) 
    {
        printf("ERROR!\n");
        printf("Usage: %s path_of_a_image\n", name);
        exit(-1);
    }

    if (-1 == access(argv[1], R_OK | F_OK) )
    {
        printf("ERROR!\n");
        printf("The file is not exist or can't be read!\n");
        exit(-1);
    }
}

int main(int argc, char** argv)
{
    char *name = "shape_eoh_cuda";

    check(argc, argv, name);

    char *path = argv[1];

    //read img
    Mat image;
    //image = imread(path, CV_LOAD_IMAGE_COLOR);
    image = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
        
    if( !image.data ) 
    {
        printf("ERROR!\n");
        printf("Can't read the file or it's not a image.\n");
        exit(-1);
    }

    //init graphic card
	int dev = 0;
	checkCudaErrors(cudaSetDevice(dev));
	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
	
	//con-currency size
	int threadPerBlock = 2*256;
	int blockPerGrid = 2*deviceProp.multiProcessorCount;
	int len_img;
	
	uchar *gray = image.ptr<uchar>(0);	
	int row = image.rows;
	int col = image.cols;
	len_img = row * col;
	
	//alloc memory on graphic card
	uchar *d_gray = NULL;
	float *d_fgray = NULL;
	float *d_gaussker = NULL;
	float *d_hist = NULL;
	int *d_hist_int = NULL;
	float *d_theta = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_gray, len_img * sizeof(uchar)));
	checkCudaErrors(cudaMalloc((void**)&d_fgray, len_img * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_gaussker, WIDTH*WIDTH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_theta, len_img * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_hist, nbins * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_hist_int, nbins * sizeof(int)));
      
    //create cublas handle
	cublasHandle_t handlet;
	cublasCreate(&handlet);
	
	//host to device
	checkCudaErrors(cudaMemcpy(d_gray, gray, len_img * sizeof(uchar), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_gaussker, gaussker, WIDTH*WIDTH*sizeof(float), cudaMemcpyHostToDevice));
    calcEOH<<<blockPerGrid, threadPerBlock, 0>>>(d_fgray, d_gray, d_theta, d_gaussker, row, col);
    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");
    calcHist<<<blockPerGrid, threadPerBlock, 0>>>(d_hist, d_hist_int, d_theta, row, col );
    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");
    checkCudaErrors(cudaMemcpy(feature, d_hist, nbins*sizeof(float), cudaMemcpyDeviceToHost));
        
	//print
    print_feature();

	//free
	cudaFree(d_gray);
	cudaFree(d_fgray);
	cudaFree(d_gaussker);
	cudaFree(d_theta);
	cudaFree(d_hist);
	cudaFree(d_hist_int);
    cublasDestroy(handlet);
	
	return 0;
}