#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <algorithm>
using namespace cv;
using namespace std;

const int ncolorspace = 3;
const int Momentnum = 3;
typedef unsigned int uint;

//feature
double *feature;

//moment of color
void calcMoment(uchar h[], double &mH, double &vH, double &sH, int row, int col, double scale)
{
    //first moment
	for (int i = 0; i < row; i++)
	    for (int j = 0; j < col; j++)
	        mH += (double)h[i * col + j] / scale;
	mH /= (double)(row * col);
	//secondary moment
	for (int i = 0; i < row; i++)
	    for (int j = 0; j < col; j++)
	        vH += pow(abs((double)h[i * col + j] / scale - mH), 2.0);
	vH /= (double)(row * col);
	vH = pow(vH, 1.0/2.0);
	//third moment
	for (int i = 0; i < row; i++)
	    for (int j = 0; j < col; j++)
	        sH += pow(abs((double)h[i * col + j]/scale - mH), 3.0);
	sH /= (double)(row * col);
	sH = pow(sH, 1.0/3.0);    
}

//calc moment feature
void calc_moment_feature(uchar h[], uchar s[], uchar v[], int row, int col)
{
    //use hsv
    
	//calc h channel
	double mH = 0.0, vH = 0.0, sH = 0.0;
	calcMoment(h, mH, vH, sH, row, col, 180.0);
	
	//calc s channel
	double mS = 0.0, vS = 0.0, sS = 0.0;
	calcMoment(s, mS, vS, sS, row, col, 255.0);
	
	//calc v channel
	double mV = 0.0, vV = 0.0, sV = 0.0;
	calcMoment(v, mV, vV, sV, row, col, 255.0);
	
	//set the value
	feature[0]=mH;feature[3]=vH;feature[6]=sH;
	feature[1]=mS;feature[4]=vS;feature[7]=sS;
	feature[2]=mV;feature[5]=vV;feature[8]=sV;
}

void print_feature()
{
    for (int i = 0; i < ncolorspace * Momentnum; i++) 
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
    char *name = "color_moment";

    check(argc, argv, name);

    feature = (double *)calloc(ncolorspace * Momentnum, sizeof(double));
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

    //change to hsv
    cvtColor(image, image_hsv, COLOR_BGR2HSV );
	//split to hsv matrix
	vector<Mat> bgr_planes;
	bgr_planes.clear();
	split( image_hsv, bgr_planes);

	uchar *h = bgr_planes[0].ptr<uchar>(0);	
	uchar *s = bgr_planes[1].ptr<uchar>(0);	
	uchar *v = bgr_planes[2].ptr<uchar>(0);	

	int row = bgr_planes[0].rows;
	int col = bgr_planes[0].cols;

	calc_moment_feature(h, s, v, row, col);

	//print
	print_feature();

	free(feature);
	return 0;
}
