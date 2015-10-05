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
#define nbins 30
typedef unsigned int uint;

//feature
double feature[nbins];

//calc EOH feature
void calcEOHFeature(uchar gray[], int hist[], int row, int col)
{
    //init
    memset(hist, 0, nbins * sizeof(int));
    int size = row * col;
    double *dg = new double[size];
    for (int i = 0; i < size; i++)
        *(dg+i) = (double)*(gray+i);
    Mat src(row, col, CV_64F, dg);
    Mat dst(row, col, CV_64F);
    
    //set gaussian filter
    int width = 5;
    Size ksize(width, width);
    double sigma = 1.0;//x,y
    GaussianBlur(src, dst, ksize, sigma, sigma);
    
    //Canny
    double kx[9] = {0,0,0,0,-1,1,0,-1,1};
    double ky[9] = {0,0,0,0,1,1,0,-1,-1};
    Mat kerx(3,3,CV_64F,kx);
    Mat kery(3,3,CV_64F,ky);
    
    //set conv params
    Point anchor = Point(-1,-1);
    double delta = 0;
    int ddepth= -1;
    
    Mat Gx, Gy;
    filter2D(dst, Gx, ddepth , kerx, anchor, delta, BORDER_DEFAULT );
    filter2D(dst, Gy, ddepth , kery, anchor, delta, BORDER_DEFAULT );
    
    //set threshold and calc feature
    double threshold = 12.0;
    for (int i = 0; i < row; i++)
    {
        double *pY = Gy.ptr<double>(i);
        double *pX = Gx.ptr<double>(i);
        for (int j = 0; j < col; j++)
        {
            double ty = *(pY+j);
            double tx = *(pX+j);
            int tmp;
            if (sqrt(ty*ty+tx*tx) > threshold)
            {
                double theta = atan(ty/tx) * 180.0 / PI + 180.0;
                tmp = (int)theta / (180 / nbins);
                tmp %= nbins;
                hist[tmp]++;
            }
        }
    }
    
    //delete [] dg; dg = 0;
}

//calc feature shape eoh
void calc_feature_shape_eoh(uchar gray[], int row, int col)
{
    int hist[nbins];
    double size = (double)row * (double)col;
    calcEOHFeature(gray, hist, row, col);
    
    for (int i = 0; i < nbins; i++)
        feature[i] = (double)hist[i]/size ;/// sqrt(sum);//(double)hist[i] / size;
}

void print_feature()
{
    for (int i = 0; i < nbins; i++) 
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
	memset(feature, 0, nbins * sizeof(double));
	uchar *gray = image.ptr<uchar>(0);	

	int row = image.rows;
	int col = image.cols;
		
	calc_feature_shape_eoh(gray, row, col);

	//print
	print_feature();

	return 0;
}