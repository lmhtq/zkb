#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <algorithm>
#include <unistd.h>
#include <fcntl.h>
using namespace cv;
using namespace std;

const int nbins = 32;
typedef unsigned int uint;

//feature
uint *feature;

//calc histogram feature
void calc_histogram_feature(uchar b[], uchar g[], uchar r[], int row, int col)
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			feature[ b[i * col + j] / 8 ]++;
		}
	}
	
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			feature[ g[i * col + j] / 8 + nbins]++;
		}
	}
	
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			feature[ r[i * col + j] / 8 + nbins * 2]++;
		}
	}
}

void print_feature()
{
	for (int i = 0; i < nbins * 3; i++) 
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
	char *name = "color_histogram";

	check(argc, argv, name);

	feature = (uint *)calloc(nbins * 3, sizeof(uint));
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

	//split BGR matrix
	vector<Mat> bgr_planes;
	bgr_planes.clear();
	split( image, bgr_planes);

	uchar *b = bgr_planes[0].ptr<uchar>(0);	
	uchar *g = bgr_planes[1].ptr<uchar>(0);	
	uchar *r = bgr_planes[2].ptr<uchar>(0);	

	int row = bgr_planes[0].rows;
	int col = bgr_planes[0].cols;

	calc_histogram_feature(b, g, r, row, col);

	print_feature();
	
	free(feature);
	return 0;
}
