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

using namespace cv;
using namespace std;

const int nbins = 32;

//feature
int *h_hist;

extern "C"

//calc histogram
__global__ void
calc_Histogram_GPU(const uchar *pB, const uchar *pG, const uchar *pR, int *pHist, int len_img)
{
	__shared__ int tmp[nbins * 3];
	tmp[threadIdx.x] = 0;
	__syncthreads();

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	while (i < len_img)
	{
		atomicAdd(&tmp[pB[i] / 8], 1);
		atomicAdd(&tmp[pG[i] / 8 + nbins], 1);
		atomicAdd(&tmp[pR[i] / 8 + nbins + nbins], 1);
		i += stride;
	}
	__syncthreads();
	if (threadIdx.x < nbins * 3)
		atomicAdd(&pHist[threadIdx.x], tmp[threadIdx.x]);
}

void print_feature()
{
	for (int i = 0; i < nbins * 3; i++) 
	{
		printf("%d ", h_hist[i]);
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
	char *name = "color_histogram_cuda";

	check(argc, argv, name);

	char *path = argv[1];

	//read img
	Mat image;
	image = imread(path, CV_LOAD_IMAGE_COLOR);

	if( !image.data ) 
	{
		printf("ERROR!\n");
		printf("Can't read the file or it's not a image.\n");
		exit(-1);
	}
    
    //init feature array
   	h_hist = (int*)calloc(nbins * 3, sizeof(int));

	//init graphic card
	int dev = 0;
	checkCudaErrors(cudaSetDevice(dev));
	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
	
	//con-currency size
	int threadPerBlock = 256;
	int blockPerGrid = deviceProp.multiProcessorCount;
	int size_img;
	int size_hist;
	int len_hist;
	int len_img;

	//split the image
	vector<Mat> bgr_planes;
	bgr_planes.clear();
	split(image, bgr_planes);
	len_img = bgr_planes[0].rows * bgr_planes[0].cols;

	uchar *pB = bgr_planes[0].ptr<uchar>(0);
	uchar *pG = bgr_planes[1].ptr<uchar>(0);
	uchar *pR = bgr_planes[2].ptr<uchar>(0);
	
	//set the memory size
	len_hist = nbins * 3;
	size_hist = len_hist * sizeof(int);
	size_img = len_img * sizeof(uchar);

    //alloc mem on graphic card
	int *d_hist = NULL;
	uchar *d_imgB = NULL;
	uchar *d_imgG = NULL;
	uchar *d_imgR = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_hist, size_hist));
	checkCudaErrors(cudaMalloc((void**)&d_imgB, size_img));
	checkCudaErrors(cudaMalloc((void**)&d_imgG, size_img));
	checkCudaErrors(cudaMalloc((void**)&d_imgR, size_img));

	
    //host to device
	checkCudaErrors(cudaMemcpy(d_imgB, pB, size_img, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_imgG, pG, size_img, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_imgR, pR, size_img, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_hist, h_hist, size_hist, cudaMemcpyHostToDevice));
    
    //calc on GPU
    calc_Histogram_GPU<<<blockPerGrid, threadPerBlock, 0>>>(d_imgB, d_imgG, d_imgR, d_hist, len_img);
    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");

    //device to host
    checkCudaErrors(cudaMemcpy(h_hist, d_hist, size_hist, cudaMemcpyDeviceToHost));

    //print feature
 	print_feature();

    //free
	cudaFree(d_imgB);
	cudaFree(d_imgG);
	cudaFree(d_imgR);
	cudaFree(d_hist);
	cudaDeviceReset();
	free(h_hist);
	return 0;
}
