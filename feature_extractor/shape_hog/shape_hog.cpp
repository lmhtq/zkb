#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <algorithm>
#include <queue>
using namespace cv;
using namespace std;

#define PI 3.14159265
#define nbins 9
#define nH 4 //divide height to nH parts
#define nW 3 //fivide weight to nW parts
#define nC 2 //Block:nC*nC parts
#define nB 6 //nB Blocks,(nH -(nC-1)) * (nW -(nC-1))

typedef unsigned int uint;

//feature
double feature[nB * nbins];

//calc HOG feature
void calcHOGFeature(uchar gray[], double hist[], int row, int col)
{
    //init
    memset(hist, 0, nB * nbins * sizeof(double));
    int size = row * col;
    double *dg = new double[size];
    for (int i = 0; i < size; i++)
        *(dg+i) = (double)*(gray+i);
    Mat src(row, col, CV_64F, dg);
    //Mat dst(row, col, CV_64F);
    
    //set gaussian filter paramters
    //int width = 5;
    //Size ksize(width, width);
    //double sigma = 1.0;//x,y同
    //GaussianBlur(src, dst, ksize, sigma, sigma);
    
    //filter conv kernel
    double kx[3] = {-1,0,1};
    double ky[3] = {-1,0,1};
    Mat kerx(1,3,CV_64F,kx);
    Mat kery(3,1,CV_64F,ky);
    
    //set conv params
    Point anchor = Point(-1,-1);
    double delta = 0;
    int ddepth= -1;
    
    Mat Gx, Gy;
    filter2D(src, Gx, ddepth , kerx, anchor, delta, BORDER_DEFAULT );
    filter2D(src, Gy, ddepth , kery, anchor, delta, BORDER_DEFAULT );
    
    //calc feature（weight oriented histogram）
    //double threshold = 12.0;
    double histtmp[nH][nW][nbins];
    memset(histtmp, 0,nH*nW*nbins*sizeof(double));
    int Hstep = row / nH;
    row = Hstep * nH;
    int Wstep = col / nW;
    col = Wstep * nW;
    for (int i = 0; i < row; i++)
    {
        double *pY = Gy.ptr<double>(i);
        double *pX = Gx.ptr<double>(i);
        for (int j = 0; j < col; j++)
        {
            double ty = *(pY+j);
            double tx = *(pX+j);
            int tmp;
            double theta = atan(ty/tx) * 180.0 / PI + 180.0;
            tmp = (int)theta / (180 / nbins);
            tmp %= nbins;
            histtmp[i/Hstep][j/Wstep][tmp]+=sqrt(ty*ty+tx*tx);
            
        }
    }
    
    //merge Block
    for (int i = 0; i < nH-(nC-1); i++)
    {
        for (int j = 0; j < nW-(nC-1); j++)
        {
            int iB = i * (nW-(nC-1)) + j;
            for (int cr = 0; cr < nC-1; cr++)
            {
                for (int cc = 0; cc < nC-1; cc++)
                {
                    //double sum = 0;
                    for (int k = 0; k < nbins; k++)
                        hist[iB * nbins +k]=histtmp[i+cr][j+cc][k];
                    //for (int k = 0; k < nbins; k++)
                    //    hist[iB * nbins +k] = histtmp[i+cr][j+cc][k] / sum ;                       
                }
            }
            
        }
    }
    
    //delete [] dg; dg = 0;
}

//calc feature shape hog
void calc_feature_shape_hog(uchar gray[], int row, int col)
{
    double hist[nB * nbins];
    double size = (double)row * (double)col;
    calcHOGFeature(gray, hist, row, col);
    for (int i = 0; i < nB * nbins; i++)
        feature[i] = hist[i] / size;
}

void print_feature()
{
    for (int i = 0; i < nB * nbins; i++) 
    {
        printf("%lf ", feature[i]);
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
    char *name = "shape_hog";

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
	memset(feature, 0, nB * nbins * sizeof(double));
	uchar *gray = image.ptr<uchar>(0);	

	int row = image.rows;
	int col = image.cols;
		
	calc_feature_shape_hog(gray, row, col);

	//print
	print_feature();

	return 0;
}