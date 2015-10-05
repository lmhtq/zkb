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

#define nbins 32
#define ntype 2

//feature
int feature[ nbins * ntype];

extern "C"

//calc bgr2hsv
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
                
        v[i] = (float)maxtmp;
        s[i] = maxtmp != 0 ? 1.0f - (float)mintmp / (float)maxtmp : 0;
        
        if (maxtmp == mintmp)
            h[i] = 0.0f;
        else if (maxtmp == r[i])
            h[i] = ( (float)g[i] - (float)b[i] ) / ( (float)maxtmp - (float)mintmp) * 60.0f;
        else if (maxtmp == g[i])
            h[i] = 120.0f + ( (float)b[i] - (float)r[i] ) / ( (float)maxtmp - (float)mintmp) * 60.0f;
        else if (maxtmp == b[i])
            h[i] = 240.0f + ( (float)r[i] - (float)g[i] ) / ( (float)maxtmp - (float)mintmp) * 60.0f;
        if (h[i] < 0)
            h[i] += 360.0f;
        
        //H:0-360 S:0-1 V:0-1
        v[i] /= 255.0f;
        
        i += stride;
    }
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
}

//calc feature coherence vector
//TODO: wrong
__global__ void
calc_feature_coherence_vector_cuda(int *quantified, int *fea, int *label, int *labelp, int *labelcnt, int row, int col)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int len  = row * col;
	int T = len / 500;//threshold
	__shared__ int feashared[nbins * ntype];
	if (threadIdx.x < nbins * ntype)
	{
	    feashared[threadIdx.x] = 0;
	    fea[threadIdx.x] = 0;
	   
	}__syncthreads();
	//set initial label
	while (i < len)
	{
	    label[i] = 0;
	    labelp[i] = i;
	    labelcnt[i] = 0;
	    i += stride;
	}
	__syncthreads();
	
	//set uncompleted label
	int uncomplete = 0;
	while (uncomplete)
	{
	    uncomplete = 0;
	    
	    i = blockDim.x * blockIdx.x + threadIdx.x;
	    while (i < len)
	    {
	        int ri = i / col;
	        int cj = i % col;
	        if (ri == 0 && cj > 0)
	        {
	            if ( quantified[cj] == quantified[cj-1] && label[cj] > labelp[cj-1])
	            {
	                atomicAdd( &label[cj], labelp[cj-1] ) ;
	                uncomplete = 1;
	            }
	        }
	        else if (ri > 0 && cj == 0)
	        {
	            if ( quantified[ri * col] == quantified[ (ri-1)*col + 1] && label[ri*col] > labelp[ (ri-1)*col+1 ] )
	            {
	                atomicAdd( &label[ ri*col ] , labelp[ (ri-1)*col+1 ] );
	                uncomplete = 1;
	            }
	            if ( quantified[ri * col] == quantified[ (ri-1)*col] && label[ri*col] > labelp[ (ri-1)*col ] )
	            {
	                atomicAdd( &label[ ri*col ], labelp[ (ri-1)*col ] );
	                uncomplete = 1;
	            }
	            
	        }
	        else if (ri > 0 && cj == col )
	        {
	            if ( quantified[ri*col+cj] == quantified[ri*col+cj-1] && label[ri*col+cj] > labelp[ri*col+cj-1] )
	            {
	                atomicAdd( &label[ri*col+cj], labelp[ri*col+cj-1] );
	                uncomplete = 1;
	            }
	            if ( quantified[ri*col+cj] == quantified[(ri-1)*col+cj] && label[ri*col+cj] > labelp[(ri-1)*col+cj-1] )
	            {
	            
	                atomicAdd( &label[ri*col+cj], labelp[(ri-1)*col+cj-1]);
	                uncomplete = 1;    
	            }
	            if ( quantified[ri*col+cj] == quantified[(ri-1)*col+cj-1] && label[ri*col+cj] > labelp[(ri-1)*col+cj-1] )
	            {
	                atomicAdd (&label[ri*col+cj], labelp[(ri-1)*col+cj-1] );
	                uncomplete = 1;
	            }
	        }
	        else
	        {
	            if ( quantified[ri*col+cj] == quantified[ri*col+cj-1] && label[ri*col+cj] > labelp[ri*col+cj-1] )
	            {
	                atomicAdd( &label[ri*col+cj], labelp[ri*col+cj-1] );
	                uncomplete = 1;
	            }
	            if ( quantified[ri * col] == quantified[ (ri-1)*col + 1] && label[ri*col] > labelp[ (ri-1)*col+1 ] )
	            {
	                atomicAdd( &label[ ri*col ], labelp[ (ri-1)*col+1 ] );
	                uncomplete = 1;
	            }
	            if ( quantified[ri*col+cj] == quantified[(ri-1)*col+cj] && label[ri*col+cj] > labelp[(ri-1)*col+cj-1] )
	            {
	                atomicAdd ( &label[ri*col+cj], labelp[(ri-1)*col+cj-1] );
	                uncomplete = 1;
	            }
	            if ( quantified[ri*col+cj] == quantified[(ri-1)*col+cj-1] && label[ri*col+cj] > labelp[(ri-1)*col+cj-1] )
	            {
	                atomicAdd ( &label[ri*col+cj], labelp[(ri-1)*col+cj-1] );
	                uncomplete = 1;   
	            }
	        }
	        
	        i += stride;
	        
	    }
	    __syncthreads();
	    i = blockDim.x * blockIdx.x + threadIdx.x;
	    while (i < len)
	    {
	        labelp[i] = label[i];
	        label[i] = 0;
	        i += stride;
	    }
	    __syncthreads();
	}
	
	i = blockDim.x * blockIdx.x + threadIdx.x;
	while (i < len)
	{
	    atomicAdd(&labelcnt[labelp[i]],1);
	    i += stride;
	}
	__syncthreads();
	
	i = blockDim.x * blockIdx.x + threadIdx.x;
	while (i < len)
	{
	    if (labelcnt[i] >= T)
	        atomicAdd(&feashared[2*quantified[i]], labelcnt[i]);
	    else if (labelcnt[i] > 0)
	        atomicAdd(&feashared[2*quantified[i]+1], labelcnt[i]);
	    i += stride;
	}
    __syncthreads();
    if (threadIdx.x < nbins * ntype)
        atomicAdd(&fea[threadIdx.x], feashared[threadIdx.x]);
}

void print_feature()
{
	for (int i = 0; i < nbins * ntype; i++) 
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


int main(int argc, char **argv)
{
	char *name = "color_coherence_vector_cuda";

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

	//init
   	memset(feature, 0, nbins * ntype * sizeof(float));

	//init graphic card
	int dev = 0;
	checkCudaErrors(cudaSetDevice(dev));
	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
		
	//con-currency size
	int threadPerBlock = 2 * 256;
	int blockPerGrid = 2 * deviceProp.multiProcessorCount;
	int size_img;
	int size_fea;
	int len_fea;
	int len_img;

	//split image
	vector<Mat> bgr_planes;
	bgr_planes.clear();
	split(image, bgr_planes);

	len_img = bgr_planes[0].rows * bgr_planes[0].cols;
	uchar *pB = bgr_planes[0].ptr<uchar>(0);
	uchar *pG = bgr_planes[1].ptr<uchar>(0);
	uchar *pR = bgr_planes[2].ptr<uchar>(0);
	
	//size
	len_fea = nbins * ntype;
	size_fea = len_fea * sizeof(int);
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
	int *d_fea = NULL;
	int *d_label = NULL;
	int *d_labelp = NULL;
	int *d_labelcnt = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_quantified, size_imgi ));
	checkCudaErrors(cudaMalloc((void**)&d_imgB, size_img ));
	checkCudaErrors(cudaMalloc((void**)&d_imgG, size_img ));
	checkCudaErrors(cudaMalloc((void**)&d_imgR, size_img ));
	checkCudaErrors(cudaMalloc((void**)&d_imgH, size_imgd ));
	checkCudaErrors(cudaMalloc((void**)&d_imgS, size_imgd ));
	checkCudaErrors(cudaMalloc((void**)&d_imgV, size_imgd ));
    checkCudaErrors(cudaMalloc((void**)&d_fea, size_fea ));
    checkCudaErrors(cudaMalloc((void**)&d_label, len_img * sizeof(int) ));
    checkCudaErrors(cudaMalloc((void**)&d_labelp, len_img * sizeof(int) ));
    checkCudaErrors(cudaMalloc((void**)&d_labelcnt, len_img * sizeof(int) ));

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
    calc_feature_coherence_vector_cuda<<<blockPerGrid, threadPerBlock, 0>>>(d_quantified, d_fea, d_label, d_labelp, d_labelcnt, row, col);
    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");
    
    //device to host
    checkCudaErrors(cudaMemcpy(feature, d_fea, size_fea, cudaMemcpyDeviceToHost));
    
    //print
    print_feature();

    //free
	cudaFree(d_imgB);
	cudaFree(d_imgG);
	cudaFree(d_imgR);
	cudaFree(d_imgH);
	cudaFree(d_imgS);
	cudaFree(d_imgV);
	cudaFree(d_label);
	cudaFree(d_labelp);
	cudaFree(d_labelcnt);
	cudaFree(d_fea);
	
	return 0;
}
