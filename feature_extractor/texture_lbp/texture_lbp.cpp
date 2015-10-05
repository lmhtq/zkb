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

const int nbins = 256;
typedef unsigned int uint;

//feature
uint *feature;

//calc LBP feature
void calc_LBP_feature(uchar gray[], int hist[], int row, int col)
{
    memset(hist, 0, nbins * sizeof(int));
    for (int i = 1; i < row-1; i++)
    {
        for (int j = 1; j < col-1; j++)
        {
            int tmp = 0;
            int anchor = gray[i * col + j];
            if (gray[(i-1)*col+j-1] > anchor)
                tmp+=128;
            if (gray[(i-1)*col+j] > anchor)
                tmp+=64;
            if (gray[(i-1)*col+j+1] > anchor)
                tmp+=32;
            if (gray[(i)*col+j+1] > anchor)
                tmp+=16;
            if (gray[(i+1)*col+j+1] > anchor)
                tmp+=8;
            if (gray[(i+1)*col+j] > anchor)
                tmp+=4;
            if (gray[(i+1)*col+j-1] > anchor)
                tmp+=2;
            if (gray[(i)*col+j-1] > anchor)
                tmp+=1;
            //128 64 32
            //1      16
            //2   4   8
            hist[tmp]++;
        }
    }
}

//calc feature of texture lbp
void calc_feature_texture_LBP(uchar gray[], int row, int col)
{
    int hist[nbins];
    calc_LBP_feature(gray, hist, row, col);
    for (int i = 0; i < nbins; i++)
        feature[i] = (uint)hist[i];
    
}

void print_feature()
{
    for (int i = 0; i < nbins; i++) 
    {
        printf("%u ", feature[i]);
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
    char *name = "texture_lbp";

    check(argc, argv, name);

    feature = (uint *)calloc(nbins, sizeof(uint));
    char *path = argv[1];

    //read img
    Mat image, image_hsv;
    //image = imread(path, CV_LOAD_IMAGE_COLOR);
    image = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
	
	if( !image.data ) 
    {
        printf("ERROR!\n");
        printf("Can't read the file or it's not a image.\n");
        exit(-1);
    }    
	
	uchar *gray = image.ptr<uchar>(0);	

	int row = image.rows;
	int col = image.cols;
		
	calc_feature_texture_LBP(gray, row, col);

	//print
	print_feature();

	free(feature);
	return 0;
}
