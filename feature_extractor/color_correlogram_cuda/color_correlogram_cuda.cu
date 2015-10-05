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
const int Dmax = 3;

//feature
float *feature;

extern "C"

//transfer bgr to hsv
__global__ void 
bgr2hsv(uchar *b, uchar *g, uchar *r, float *h, float *s, float *v, int row, int col)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
    int len = row * col;
    uchar maxtmp;
    uchar mintmp;
    while (i < len)
    {
        maxtmp = b[i] > g[i] ? b[i] : g[i];
        maxtmp = maxtmp > r[i] ? maxtmp : r[i];
        mintmp = b[i] < g[i] ? b[i] : g[i];
        mintmp = mintmp < r[i] ? mintmp : r[i];
        float maxi = (float)maxtmp;
        float mini = (float)mintmp;
        v[i] = maxi;
        s[i] = (maxi - mini) / maxi;
        if (fabs(maxi) < 1e-6)
            s[i] = 0;
        if (fabs(maxi - mini) < 1e-6)
            h[i] = 0;
        else if (fabs(maxi - (float)r[i]) < 1e-6)
        {
            h[i] = (float)(g[i] - b[i]) / (float)(maxi - mini) * 60.0;
        }
        else if (fabs(maxi - (float)g[i]) < 1e-6)
        {
            h[i] = 120.0 + (float)(b[i] - r[i]) / (float)(maxi - mini) * 60.0;
        }
        else if (fabs(maxi - (float)b[i]) < 1e-6)
        {
            h[i] = 240.0 + (float)(r[i] - g[i]) / (float)(maxi - mini) * 60.0;
        }
        if (h[i] < 0)
            h[i] += 360.0;
                
        //H:0-360 S:0-1 V:0-1
        v[i] /= 255.0;
        
        i += stride;
    }
    __syncthreads();
}

//quantify HSV
//based on the paper《基于色彩量化及索引的图像检索》
__global__ void 
Quantify(int *quantified, float *h, float *s, float *v, int row, int col)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
    int len = row * col;
    while (i < len)
    {
        if (v[i] <= 0.1)
            quantified[i] = 0;
        else if (v[i] > 0.1 && v[i] <= 0.4 && s[i] <= 0.1)
            quantified[i] = 1;
        else if (v[i] > 0.4 && v[i] <= 0.7 && s[i] <= 0.1)
            quantified[i] = 2;
        else if (v[i] > 0.7 && v[i] <= 1.0 && s[i] <= 0.1)
            quantified[i] = 3;
        else
        {
            if (s[i] > 0.1 && s[i] <= 0.5 && v[i] > 0.1 && v[i] <= 0.5)
            {
                if (h[i] > 20 && h[i] <= 45)
                    quantified[i] = 4;
                else if (h[i] > 45 && h[i] <= 75)
                    quantified[i] = 5;
                else if (h[i] > 75 && h[i] <= 155)
                    quantified[i] = 6;
                else if (h[i] > 155 && h[i] <= 210)
                    quantified[i] = 7;
                else if (h[i] > 210 && h[i] <= 270)
                    quantified[i] = 8;
                else if (h[i] > 270 && h[i] <= 330)
                    quantified[i] = 9;
                else
                    quantified[i] = 10;
            }
            else if (s[i] > 0.1 && s[i] <= 0.5 && v[i] > 0.5 && v[i] <= 1.0)
            {
                if (h[i] > 20 && h[i] <= 45)
                    quantified[i] = 11;
                else if (h[i] > 45 && h[i] <= 75)
                    quantified[i] = 12;
                else if (h[i] > 75 && h[i] <= 155)
                    quantified[i] = 13;
                else if (h[i] > 155 && h[i] <= 210)
                    quantified[i] = 14;
                else if (h[i] > 210 && h[i] <= 270)
                    quantified[i] = 15;
                else if (h[i] > 270 && h[i] <= 330)
                    quantified[i] = 16;
                else
                    quantified[i] = 17;
            }
            else if (s[i] > 0.5 && s[i] <= 1.0 && v[i] > 0.1 && v[i] <= 0.5)
            {
                if (h[i] > 20 && h[i] <= 45)
                    quantified[i] = 18;
                else if (h[i] > 45 && h[i] <= 75)
                    quantified[i] = 19;
                else if (h[i] > 75 && h[i] <= 155)
                    quantified[i] = 20;
                else if (h[i] > 155 && h[i] <= 210)
                    quantified[i] = 21;
                else if (h[i] > 210 && h[i] <= 270)
                    quantified[i] = 22;
                else if (h[i] > 270 && h[i] <= 330)
                    quantified[i] = 23;
                else
                    quantified[i] = 24;
            }
            else if (s[i] > 0.5 && s[i] <= 1.0 && v[i] > 0.5 && v[i] <= 1.0)
            {
                if (h[i] > 20 && h[i] <= 45)
                    quantified[i] = 25;
                else if (h[i] > 45 && h[i] <= 75)
                    quantified[i] = 26;
                else if (h[i] > 75 && h[i] <= 155)
                    quantified[i] = 27;
                else if (h[i] > 155 && h[i] <= 210)
                    quantified[i] = 28;
                else if (h[i] > 210 && h[i] <= 270)
                    quantified[i] = 29;
                else if (h[i] > 270 && h[i] <= 330)
                    quantified[i] = 30;
                else
                    quantified[i] = 31;
            }
        }
    
        i += stride;
    }
    __syncthreads();
}

//calc correlogram feature
__global__ void
calc_featue_correlogram(int *quantified, float *fea, int row, int col, int *feai)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	__shared__ int tmp[nbins * Dmax];
	tmp[threadIdx.x] = 0;
    feai[threadIdx.x] = 0;
    __syncthreads();
    
    int len = row * col;
    while (i < len)
    {
        int ri = i / col;//row id
        int cj = i % col;//colum id
        for (int k = 0 ; k < Dmax; k++)
        {
            if (ri - k < 0 || cj - k < 0 || ri + k >= row || cj + k >= col)
                continue;
            int cnt = 0;
            for (int ii = ri - k; ii <= ri + k; ii++)
                for (int jj = cj - k; jj <= cj + k; jj++)
                    if ( quantified[ii * col + jj ] == quantified[i] )
                        cnt++;
            atomicAdd(&tmp[quantified[i] * Dmax + k] , cnt );
            //atomicAdd(&feai[quantified[i] * Dmax + k] , cnt );
        }
        i += stride;
    }
    __syncthreads();
	
    if (threadIdx.x < nbins * Dmax)
    {
        atomicAdd(&feai[threadIdx.x], tmp[threadIdx.x]);
    }
    
    if (threadIdx.x < nbins * Dmax)
    {
        int k = 2*(threadIdx.x % Dmax)+1;
        fea[threadIdx.x] = (float)feai[threadIdx.x] / ((float)(k * k * row * col));
    }
    
}

void print_feature()
{
	for (int i = 0; i < nbins * Dmax; i++) 
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
	char *name = "color_correlogram_cuda";

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
    feature = (float *)calloc(nbins * Dmax, sizeof(float));
   	
	//init graphic card
	int dev = 0;
	checkCudaErrors(cudaSetDevice(dev));
	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
	
	//con-currency size
	int threadPerBlock = 256;
	int blockPerGrid = deviceProp.multiProcessorCount;
	int size_img;
	int size_fea;
	int len_fea;
	int len_img;

	//split images
	vector<Mat> bgr_planes;
	bgr_planes.clear();
	split(image, bgr_planes);

	len_img = bgr_planes[0].rows * bgr_planes[0].cols;
	uchar *pB = bgr_planes[0].ptr<uchar>(0);
	uchar *pG = bgr_planes[1].ptr<uchar>(0);
	uchar *pR = bgr_planes[2].ptr<uchar>(0);
	
	//size
	len_fea = nbins * Dmax;
	size_fea = len_fea * sizeof(float);
	size_img = len_img * sizeof(uchar);
	int size_imgd = len_img * sizeof(float);
	int size_imgi = len_img * sizeof(int);
	
    //malloc memory on graphic card
	int *d_quantified = NULL;
	uchar *d_imgB = NULL;
	uchar *d_imgG = NULL;
	uchar *d_imgR = NULL;
	float *d_imgH = NULL;
	float *d_imgS = NULL;
	float *d_imgV = NULL;
	float *d_fea = NULL;
	int *d_feai = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_quantified, size_imgi));
	checkCudaErrors(cudaMalloc((void**)&d_imgB, size_img));
	checkCudaErrors(cudaMalloc((void**)&d_imgG, size_img));
	checkCudaErrors(cudaMalloc((void**)&d_imgR, size_img));
	checkCudaErrors(cudaMalloc((void**)&d_imgH, size_imgd));
	checkCudaErrors(cudaMalloc((void**)&d_imgS, size_imgd));
	checkCudaErrors(cudaMalloc((void**)&d_imgV, size_imgd));
    checkCudaErrors(cudaMalloc((void**)&d_fea, size_fea));
    checkCudaErrors(cudaMalloc((void**)&d_feai, len_fea * sizeof(int)));

    int row = bgr_planes[0].rows;
    int col = bgr_planes[0].cols;   
    //host to device
	checkCudaErrors(cudaMemcpy(d_imgB, pB, size_img, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_imgG, pG, size_img, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_imgR, pR, size_img, cudaMemcpyHostToDevice));
		
    //calc bgr2hsv
    bgr2hsv<<<blockPerGrid, threadPerBlock, 0>>>(d_imgB, d_imgG, d_imgR, d_imgH, d_imgS, d_imgV, row, col);
    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");
    
    //quantify HSV
    Quantify<<<blockPerGrid, threadPerBlock, 0>>>(d_quantified, d_imgH, d_imgS, d_imgV, row, col);
    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");
    
    //calc feature
    calc_featue_correlogram<<<blockPerGrid, threadPerBlock, 0>>>(d_quantified, d_fea, row, col, d_feai);
    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");
    
    //device to host
    checkCudaErrors(cudaMemcpy(feature, d_fea, size_fea, cudaMemcpyDeviceToHost));
    
    //print feature
    print_feature();

	//free
    cudaFree(d_imgB);
    cudaFree(d_imgG);
    cudaFree(d_imgR);
    cudaFree(d_imgH);
    cudaFree(d_imgS);
    cudaFree(d_imgV);
    cudaFree(d_quantified);
    cudaFree(d_feai);
	cudaFree(d_fea);
	free(feature);
	return 0;
}
