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

#define ncolorspace 3
#define Momentnum 3
#define STREAM_NUM 1

//faeture
float feature[ ncolorspace * Momentnum];
float *B;
float *G;
float *R;
extern "C"

//calc vector-number
__global__ void 
apx(float *x, float a, int  n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	
    while (i < n)
    {
        x[i] -= a;
        i += stride;
    }
}

//calc element pow
__global__ void 
xpowa(float *x, float a, float *y, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
    while (i < n)
    {
        y[i] = powf(x[i], a);    
        i += stride;
    }
}

void print_feature()
{
	for (int i = 0; i < ncolorspace * Momentnum; i++) 
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


int main(int argc, char **argv)
{
	char *name = "color_moment_cuda";

	check(argc, argv, name);

	char *path = argv[1];

	//read img
	Mat image, image_hsv;
	image = imread(path, CV_LOAD_IMAGE_COLOR);

	if( !image.data ) 
	{
		printf("ERROR!\n");
		printf("Can't read the file or it's not a image.\n");
		exit(-1);
	}

	//init
   	memset(feature, 0, ncolorspace * Momentnum * sizeof(float));

	//init graphic card
	int dev = 0;
	checkCudaErrors(cudaSetDevice(dev));
	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
	cublasStatus_t cubStatus = CUBLAS_STATUS_SUCCESS;
	
	//con-currency size
	int threadPerBlock = 2 * 256;
	int blockPerGrid = 2 * deviceProp.multiProcessorCount;
	int size_img;
	int size_imgf;
	int size_fea;
	int len_fea;
	int len_img;

	//change to hsv
    cvtColor(image, image_hsv, COLOR_BGR2HSV );
	//split image
	vector<Mat> bgr_planes;
	bgr_planes.clear();
	split(image_hsv, bgr_planes);
    
	len_img = bgr_planes[0].rows * bgr_planes[0].cols;
	uchar *pB = bgr_planes[0].ptr<uchar>(0);
	uchar *pG = bgr_planes[1].ptr<uchar>(0);
	uchar *pR = bgr_planes[2].ptr<uchar>(0);
	B = (float*)calloc(len_img, sizeof(float));
	G = (float*)calloc(len_img, sizeof(float));
	R = (float*)calloc(len_img, sizeof(float));
	int k;
	for (k = 0; k < len_img; k++)
	    B[k] = (float)pB[k]/180.0;//scale
    for (k = 0; k < len_img; k++)
	    G[k] = (float)pG[k]/255.0;
    for (k = 0; k < len_img; k++)
	    R[k] = (float)pR[k]/255.0;
 
	//size
	len_fea = ncolorspace * Momentnum;
	size_fea = len_fea * sizeof(float);
	size_img = len_img * sizeof(uchar);
	size_imgf = len_img * sizeof(float);
	//int size_imgd = len_img * sizeof(float);
	//int size_imgi = len_img * sizeof(int);
	
    //malloc memory on graphic card
	float *d_imgB = NULL;
	float *d_imgG = NULL;
	float *d_imgR = NULL;
	float *d_imgB2 = NULL;
	float *d_imgG2 = NULL;
	float *d_imgR2 = NULL;
	float *d_imgB3 = NULL;
	float *d_imgG3 = NULL;
	float *d_imgR3 = NULL;
	float *d_fea = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_imgB, size_imgf));
	checkCudaErrors(cudaMalloc((void**)&d_imgG, size_imgf));
	checkCudaErrors(cudaMalloc((void**)&d_imgR, size_imgf));
	checkCudaErrors(cudaMalloc((void**)&d_imgB2, size_imgf));
	checkCudaErrors(cudaMalloc((void**)&d_imgG2, size_imgf));
	checkCudaErrors(cudaMalloc((void**)&d_imgR2, size_imgf));
	checkCudaErrors(cudaMalloc((void**)&d_imgB3, size_imgf));
	checkCudaErrors(cudaMalloc((void**)&d_imgG3, size_imgf));
	checkCudaErrors(cudaMalloc((void**)&d_imgR3, size_imgf));
	checkCudaErrors(cudaMalloc((void**)&d_fea, size_fea));

	cublasHandle_t handle[STREAM_NUM];
	cudaStream_t stream[STREAM_NUM];
	for (int i = 0; i < STREAM_NUM; i++)
		cudaStreamCreate(&stream[i]);
    for (int i = 0; i < STREAM_NUM; i++)
    {
        cublasCreate(&handle[i]);
		cublasSetStream(handle[i], stream[i]);
    }
    
    int j = 0;
    int row = bgr_planes[0].rows;
    int col = bgr_planes[0].cols;   
    //host to device
    cubStatus = cublasSetVector(len_img, sizeof(float), B, 1, d_imgB, 1);
	CUBLAS_ERROR_CHECK(cubStatus)
	//sum
	cublasSasum(handle[j], len_img, d_imgB, 1, &feature[0]);
	//mean
	feature[0] /= (float)len_img; 
	//vec-num
	apx<<<blockPerGrid, threadPerBlock, 0, stream[j]>>>(d_imgB, feature[0], len_img);
    //ele pow2
    xpowa<<<blockPerGrid, threadPerBlock, 0, stream[j]>>>(d_imgB, 2.0, d_imgB2, len_img );
    //sum of pow2
    cublasSasum(handle[j], len_img, d_imgB2, 1, &feature[3]);
	//2nd moment
	feature[3] /= (float)len_img; feature[3] = sqrt(feature[3]);
	//ele pow3
	xpowa<<<blockPerGrid, threadPerBlock, 0, stream[j]>>>(d_imgB, 3.0, d_imgB3, len_img );
    //sum of pow3
    cublasSasum(handle[j], len_img, d_imgB3, 1, &feature[6]);
	//3rd moment
	feature[6] /= (float)len_img; feature[6] = pow(feature[6],1.0/3.0);
	    
	//G channel
	cublasSetVector(len_img, sizeof(float), G, 1, d_imgG, 1);
	cublasSasum(handle[j], len_img, d_imgG, 1, &feature[1]);
	feature[1] /= (float)len_img;
	apx<<<blockPerGrid, threadPerBlock, 0, stream[j]>>>(d_imgG, feature[1], len_img);
    xpowa<<<blockPerGrid, threadPerBlock, 0, stream[j]>>>(d_imgG, 2.0, d_imgG2, len_img );
    cublasSasum(handle[j], len_img, d_imgG2, 1, &feature[4]);
	feature[4] /= (float)len_img; feature[4] = sqrt(feature[4]);
	xpowa<<<blockPerGrid, threadPerBlock, 0, stream[j]>>>(d_imgG, 3.0, d_imgG3, len_img );
    cublasSasum(handle[j], len_img, d_imgG3, 1, &feature[7]);
	feature[7] /= (float)len_img; feature[7] = pow(feature[7],1.0/3.0);
	    
	//R channel
	cublasSetVector(len_img, sizeof(float), R, 1, d_imgR, 1);
	cublasSasum(handle[j], len_img, d_imgR, 1, &feature[2]);
	feature[2] /= (float)len_img;	    
	apx<<<blockPerGrid, threadPerBlock, 0, stream[j]>>>(d_imgR, feature[2], len_img);
    xpowa<<<blockPerGrid, threadPerBlock, 0, stream[j]>>>(d_imgR, 2.0, d_imgR2, len_img );
    cublasSasum(handle[j], len_img, d_imgR2, 1, &feature[5]);
	feature[5] /= (float)len_img; feature[5] = sqrt(feature[5]);
	xpowa<<<blockPerGrid, threadPerBlock, 0, stream[j]>>>(d_imgR, 3.0, d_imgR3, len_img );
    cublasSasum(handle[j], len_img, d_imgR3, 1, &feature[8]);
	feature[8] /= (float)len_img; feature[8] = pow(feature[8],1.0/3.0);
	
	
	for (int j = 0; j < STREAM_NUM; j++)
	        cudaStreamSynchronize(stream[j]);
	
	for (int i = 0; i < STREAM_NUM; ++i)
        cudaStreamDestroy(stream[i]);
    
    //print
    print_feature();

    //free
	cudaFree(d_imgB);
	cudaFree(d_imgG);
	cudaFree(d_imgR);
	cudaFree(d_imgB2);
	cudaFree(d_imgG2);
	cudaFree(d_imgR2);
	cudaFree(d_imgB3);
	cudaFree(d_imgG3);
	cudaFree(d_imgR3);
	cudaFree(d_fea);
	for (int i = 0; i < STREAM_NUM; i++)
        cublasDestroy(handle[i]);
	free(B);
	free(G);
	free(R);
	return 0;
}