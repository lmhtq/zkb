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

#define nScale 5
#define nDirection 6
#define ntype 2
#define width 2
#define WIDTH 5

#define PI 3.14159265
//feature
float feature[ nScale * nDirection * ntype];
float gaborReal[nScale][nDirection][WIDTH * WIDTH];
float gaborImag[nScale][nDirection][WIDTH * WIDTH];

extern "C"

//generate Gabor filter
__global__ void
calcGabor(float *d_gaborReal, float * d_gaborImag, float *xx, float *yy, int m, int n)
{
    int M = nScale;
    int N = nDirection;
    float Uh = 0.4;
    float Ul = 0.5;
    float a = powf( (Uh/Ul), -1.0/(M-1) );
    
    float W = 0.5;
    
    float sigma_x = (a+1) * sqrtf(2.0*logf(2.0)) / (2.0 * PI * powf(a, m) * (a-1) * Ul);
    float sigma_y = 1.0 / ( 2*PI * tan(PI/2*N) * sqrtf( abs((Uh*Uh)/2.0*logf(2.0) - (1.0/2.0*PI*sigma_x)*(1.0/2.0*PI*sigma_x) )) );
    
    float theta = (float)n * PI / (float)N;
    float costheta = cosf(theta);
    float sintheta = sinf(theta);
    
    float a1 = powf(a, -m);
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
    int len = WIDTH * WIDTH;
    
    while (i < len)
    {
        int ri = i / WIDTH;
        int cj = i % WIDTH;
        xx[ i ] = a1 * ( (float)(cj-width)*costheta + (float)(ri-width)*sintheta );
        yy[ i ] = a1 * ( (float)(cj-width)*(-sintheta) + (float)(ri-width)*costheta );
        
        i += stride;
    }
    __syncthreads();
    
    i = blockDim.x * blockIdx.x + threadIdx.x;
    float a2 = a1 / (2 * PI * sigma_x * sigma_y);
    while (i < len)
    {
        //int ri = i / WIDTH;
        //int cj = i % WIDTH;
        float b = expf( -0.5f*( xx[i]*xx[i]/(sigma_x*sigma_x) - yy[i]*yy[i]/(sigma_y*sigma_y) ) );
        float tmpreal = cosf(2.0*PI*W*xx[i]);
        float tmpimag = sinf(2.0*PI*W*xx[i]);
        d_gaborReal[ i ] = a2 * b * tmpimag;
        d_gaborImag[ i ] = a2 * b * tmpreal;
        i += stride;
    }
}

//([1,2,3]-1).^2(matlab)
__global__ void
VecSubaPowN(float *d_fgray, float miu, float *d_fgraymiuN, float N, int row, int col)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
    int len = row * col;
    while (i < len)
    {
        float tmp = (float)( d_fgray[i] - miu );
        d_fgraymiuN[i] = powf( tmp, N );
        i += stride;
    }
    __syncthreads();
}

//filter2D
__global__ void
Filter2D(float *dst, float *src, uchar *gray, int row, int col, float *kernel_r, float *kernel_i, int H, int W)//保证奇数
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
    int len = row * col;
    //float ker_r[WIDTH*WIDTH], ker_i[WIDTH*WIDTH];
    while (i < len)
    {
        src[i] = (float)gray[i];
        i += stride;
    }
    __syncthreads();
    
    //for (int j = 0; j < WIDTH * WIDTH; j++)
    //{
    //    ker_r[j] = kernel_r[j];
    //    ker_i[j] = kernel_i[j];
    //}
    
    i = blockDim.x * blockIdx.x + threadIdx.x;
	while (i < len)
    {
        float tmp_r = 0, tmp_i = 0;
        int r = i / col;
        int c = i % row;
        float a_r, a_i ; 
        int j = 0;               
        for (int ki = -width; ki <= width; ki++)
        {
            for (int kj = -width; kj <= width; kj++)
            {
                a_r = a_i = 0;
                if (r + ki >= 0 && r + ki < row && c + kj >= 0 && c + kj < col)
                {                    
                    a_r = src[ (r+ki)*col+c+kj ] * kernel_r[j];
                    a_i = src[ (r+ki)*col+c+kj ] * kernel_i[j];
                }
                tmp_r += a_r;
                tmp_i += a_i;
                j++;
            }
        }
        dst[i] = sqrtf( tmp_r*tmp_r + tmp_i*tmp_i );
        i += stride;
    }
    
    __syncthreads();
}

//TODO:?different from CPU version
void print_feature()
{
    for (int i = 0; i < nScale * nDirection * ntype; i++) 
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
    char *name = "texture_gabor_cuda";

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
   	memset(feature, 0, nScale * nDirection * ntype * sizeof(float));

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
	uchar *d_imgGray = NULL;
	float *d_fgray = NULL;
	float *d_gaborReal = NULL;
	float *d_xx = NULL;
	float *d_yy = NULL;
	float *d_gaborImag = NULL;
	float *d_dst = NULL;
	float *d_fgraymiu2 = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_imgGray, len_img * sizeof(uchar)));
	checkCudaErrors(cudaMalloc((void**)&d_fgray, len_img * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_gaborReal, WIDTH*WIDTH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_gaborImag, WIDTH*WIDTH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_xx, WIDTH*WIDTH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_yy, WIDTH*WIDTH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_dst, len_img * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_fgraymiu2, len_img * sizeof(float)));
      
    //create cublas handle
	cublasHandle_t handlet;
	cublasCreate(&handlet);
	
    //host to device
    checkCudaErrors(cudaMemcpy(d_imgGray, gray, len_img * sizeof(uchar), cudaMemcpyHostToDevice));
        
    for (int im = 0; im < nScale; im++)
    {
        for (int id = 0; id < nDirection; id++)
        {
            int offset = im * nDirection + id;
            calcGabor<<<blockPerGrid, threadPerBlock, 0>>>(d_gaborReal, d_gaborImag, d_xx, d_yy , im, id);
            // check if kernel execution generated and error
            getLastCudaError("Kernel execution failed");checkCudaErrors(cudaMemcpy(gaborReal[im][id], d_gaborReal, WIDTH*WIDTH*sizeof(float), cudaMemcpyDeviceToHost));
            
            checkCudaErrors(cudaMemcpy(gaborImag[im][id], d_gaborImag, WIDTH*WIDTH*sizeof(float), cudaMemcpyDeviceToHost));
            Filter2D<<<blockPerGrid, threadPerBlock, 0>>>(d_dst, d_fgray, d_imgGray, row, col, d_gaborReal, d_gaborImag, WIDTH, WIDTH);
            // check if kernel execution generated and error
            getLastCudaError("Kernel execution failed");
            cublasSasum( handlet, len_img, d_dst, 1, &feature[offset*2] );
            feature[offset*2] /= (float)(len_img);
            VecSubaPowN<<<blockPerGrid, threadPerBlock, 0>>>(d_dst, feature[offset*2], d_fgraymiu2, 2.0, row, col);
            // check if kernel execution generated and error
            getLastCudaError("Kernel execution failed");
            cublasSasum( handlet, len_img, d_fgraymiu2, 1, &feature[offset*2+1] );
            //feature[i][offset*2+1] = sqrtf( feature[i][offset*2+1] );
                
        }
    }
	
    //print
    print_feature();

	//free
	cudaFree(d_imgGray);
	cudaFree(d_fgray);
	cudaFree(d_gaborReal);
	cudaFree(d_gaborImag);
	cudaFree(d_xx);
	cudaFree(d_yy);
	cudaFree(d_dst);
	cudaFree(d_fgraymiu2);
	cublasDestroy(handlet);
	
	return 0;
}