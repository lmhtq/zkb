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

#define nbins 256

#define PI 3.14159265
//feature
int feature[ nbins];

extern "C"

//calc lbp hist
__global__ void
calc_feature_lbp(int *d_hist, uchar *gray, int row, int col)
{
    __shared__ int tmp[nbins];
    tmp[threadIdx.x] = 0;
    d_hist[threadIdx.x] = 0;
    __syncthreads();
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
	int len = row * col;
	
	while (i < len)
	{
	    //128 64 32
        //1      16
        //2   4   8
	    int r = i / col;
	    int c = i % col;
	    int sum = 0;
	    int anchor = (int)gray[i];
	    if ( r >= 1 && r < row-1 && c >= 1 && c < col - 1)
	    {
	        if ( (int)gray[ (r-1)*col+c-1 ] > anchor )
	            sum += 128;
	        if ( (int)gray[ (r-1)*col+c ] > anchor )
	            sum += 64;
	        if ( (int)gray[ (r-1)*col+c+1 ] > anchor )
	            sum += 32;
	        if ( (int)gray[ (r)*col+c+1 ] > anchor )
	            sum += 16;
	        if ( (int)gray[ (r+1)*col+c+1 ] > anchor )
	            sum += 8;
	        if ( (int)gray[ (r+1)*col+c ] > anchor )
	            sum += 4;
	        if ( (int)gray[ (r+1)*col+c-1 ] > anchor )
	            sum += 2;
	        if ( (int)gray[ (r)*col+c-1 ] > anchor )
	            sum += 1;
	            
	    }
	    atomicAdd( &tmp[sum], 1);
	    i += stride;
	}
    __syncthreads();
    atomicAdd( &d_hist[threadIdx.x], tmp[threadIdx.x] );
}

void print_feature()
{
    for (int i = 0; i < nbins; i++) 
    {
        printf("%d ", feature[i]);
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
    char *name = "texture_lbp_cuda";

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

	//split to BGR matrix
	vector<Mat> bgr_planes;
	bgr_planes.clear();
	split( image, bgr_planes);

    //init
   	memset(feature, 0, nbins * sizeof(float));

	//init graphic card
	int dev = 0;
	checkCudaErrors(cudaSetDevice(dev));
	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
	//cublasStatus_t cubStatus = CUBLAS_STATUS_SUCCESS;
	
	//con-currency size
	int threadPerBlock = 256;
	int blockPerGrid = 2*deviceProp.multiProcessorCount;
	int len_img;
	
	uchar *gray = image.ptr<uchar>(0);	
	int row = image.rows;
	int col = image.cols;
	len_img = row * col;
	
	//malloc memory on graphic card
	uchar *d_imgGray = NULL;
	int *d_hist = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_imgGray, len_img * sizeof(uchar)));
	checkCudaErrors(cudaMalloc((void**)&d_hist, nbins * sizeof(int)));
	
    //create a cublas handle
	cublasHandle_t  handlet;
	cublasCreate(&handlet);
	
    //host to device
    checkCudaErrors(cudaMemcpy(d_imgGray, gray, len_img * sizeof(uchar), cudaMemcpyHostToDevice));
    calc_feature_lbp<<<blockPerGrid, threadPerBlock, 0>>>(d_hist, d_imgGray, row, col);
    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");
    checkCudaErrors(cudaMemcpy(feature, d_hist, nbins*sizeof(int), cudaMemcpyDeviceToHost));
    
    //print
    print_feature();

    //free
    cudaFree(d_imgGray);
	cudaFree(d_hist);
	cublasDestroy(handlet);
	
	return 0;
}