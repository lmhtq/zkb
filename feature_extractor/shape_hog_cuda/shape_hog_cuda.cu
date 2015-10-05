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


#define nbins 9
#define nH 4
#define nW 3
#define nC 2
#define nB 6

#define T 12.0f
#define PI 3.141593
//feature
float feature[nB * nbins];

extern "C"

//HOG without Gamma and gaussian
__global__ void
calcHOG(float *d_fgray, uchar *gray, int *cell_hist, float *block_hist, int row, int col)
{
    __shared__ int tmp[nH*nW*nbins];
    if (threadIdx.x < nH*nW*nbins)
        tmp[threadIdx.x] = 0;
    __syncthreads();
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
    int len = row * col;
    int row2 = row / nH * nH;
    int col2 = col / nW * nW;
    int Hstep = row2 / nH;
    int Wstep = col2 / nW;
    
    while (i < nH*nW*nbins)
    {
        cell_hist[i] = 0;
        i += stride;
    }
    __syncthreads();
    
    i = blockDim.x * blockIdx.x + threadIdx.x;
	while (i < len)
    {
        d_fgray[i] = (float)gray[i];
        i += stride;
    }
    __syncthreads();
    
    //filter
    //此处通过设置变量，省去了Gx Gy Mag
    //int fx[3] = {-1,0,1};
    //int fy[3] = {-1,0,1};
    i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < len)
    {
        int r = i / col;
        int c = i % col;
        float tmpx = 0.0f;
        float tmpy = 0.0f;
        
        if ( r > 0 && r < row-1 && c > 0 && c < col-1 )
        {
            tmpx = d_fgray[i+1] - d_fgray[i-1];
            tmpy = d_fgray[i+col]-d_fgray[i-col];
        }
        
        
        float t_theta = atan( tmpy/tmpx ) / PI * 180.0f + 180.0f;
        int ind = ( (int)t_theta / (180 / nbins) ) % nbins;
        atomicAdd( &tmp[ (r/Hstep*nW + c/Wstep) * nbins + ind ], (int)sqrtf( tmpx*tmpx + tmpy*tmpy ) );
        i += stride;
    }
    __syncthreads();
    
    if (threadIdx.x < nH*nW*nbins)
        atomicAdd( &cell_hist[threadIdx.x], tmp[threadIdx.x] );
    __syncthreads();
    //merge
    i = blockDim.x * blockIdx.x + threadIdx.x;
	while (i < nbins*nH*nW)
	{
	    int ind = i % nbins;
	    int ti = i / nbins;
	    int ir = ti / nW;
	    int ic = ti % nW;
	    if (ir < nH -1 && ic < nW -1)
	    {
	        int index = (ir*(nW-1)+ic) * nbins + ind;
	        block_hist[ index ] = (float)( cell_hist[ index ] + cell_hist[ index + nbins] + 
	                                cell_hist[ index + nW*nbins ] + cell_hist[ index + (nW+1)*nbins ] )
	                                / float(len);
	    }
	    i += stride;
	}
}
//TODO:?different from CPU versions
void print_feature()
{
    for (int i = 0; i < nB * nbins; i++) 
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
    char *name = "shape_hog_cuda";

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

	//init
	memset(feature, 0, nB * nbins * sizeof(float));
    
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
	int *d_cellhist = NULL;
	float *d_blockhist = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_gray, len_img * sizeof(uchar)));
	checkCudaErrors(cudaMalloc((void**)&d_fgray, len_img * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_cellhist, nH*nW*nbins * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_blockhist, nB*nbins* sizeof(float)));
    
    //create cublas handle  
    cublasHandle_t handlet;
	cublasCreate(&handlet);
	
	checkCudaErrors(cudaMemcpy(d_gray, gray, len_img * sizeof(uchar), cudaMemcpyHostToDevice));
    calcHOG<<<blockPerGrid, threadPerBlock, 0>>>(d_fgray, d_gray, d_cellhist, d_blockhist, row, col);
    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaMemcpy(feature, d_blockhist, nB*nbins*sizeof(float), cudaMemcpyDeviceToHost));
    
    //print
    print_feature();    
	
	//free
	cudaFree(d_gray);
	cudaFree(d_fgray);
	cudaFree(d_cellhist);
	cudaFree(d_blockhist);
	cublasDestroy(handlet);
	return 0;
}
