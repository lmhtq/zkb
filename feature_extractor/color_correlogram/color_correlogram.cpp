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

//#define DEBUG

const int nbins = 32;
const int Dmax = 3;
typedef unsigned int uint;

//correlogram feature
double *feature;

//transfer bgr color space to hsv color space
void bgr2hsv(uchar b[], uchar g[], uchar r[], double h[], double s[], double v[], int row, int col)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            int tmp = i * col + j;
            double maxtmp = (double)max(r[tmp], max(g[tmp], b[tmp]));
            double mintmp = (double)min(r[tmp], min(g[tmp], b[tmp]));
            v[tmp] = maxtmp;
            s[tmp] = (maxtmp - mintmp) / maxtmp;
            if (abs(maxtmp) < 1e-6)
                s[tmp] = 0;
            if (abs(maxtmp - mintmp) < 1e-6)
                h[tmp] = 0;
            else if (abs(maxtmp - r[tmp]) < 1e-6)
            {
                h[tmp] = (g[tmp] - b[tmp]) / (maxtmp - mintmp) * 60.0;
            }
            else if (abs(maxtmp - g[tmp]) < 1e-6)
            {
                h[tmp] = 120.0 + (b[tmp] - r[tmp]) / (maxtmp - mintmp) * 60.0;
            }
            else if (abs(maxtmp - b[tmp]) < 1e-6)
            {
                h[tmp] = 240.0 + (r[tmp] - g[tmp]) / (maxtmp - mintmp) * 60.0;
            }
            if (h[tmp] < 0)
                h[tmp] += 360.0;
                
            //H:0-360 S:0-1 V:0-1
            v[tmp] /= 255.0;
        }
    }
}

//quantify HSV
//based on the paper《基于色彩量化及索引的图像检索》
void Quantify(int quantified[], double h[], double s[], double v[], int row, int col)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            int tmp = i * col + j;
            if (v[tmp] <= 0.1)
                quantified[tmp] = 0;
            else if (v[tmp] > 0.1 && v[tmp] <= 0.4 && s[tmp] <= 0.1)
                quantified[tmp] = 1;
            else if (v[tmp] > 0.4 && v[tmp] <= 0.7 && s[tmp] <= 0.1)
                quantified[tmp] = 2;
            else if (v[tmp] > 0.7 && v[tmp] <= 1.0 && s[tmp] <= 0.1)
                quantified[tmp] = 3;
            else
            {
                if (s[tmp] > 0.1 && s[tmp] <= 0.5 && v[tmp] > 0.1 && v[tmp] <= 0.5)
                {
                    if (h[tmp] > 20 && h[tmp] <= 45)
                        quantified[tmp] = 4;
                    else if (h[tmp] > 45 && h[tmp] <= 75)
                        quantified[tmp] = 5;
                    else if (h[tmp] > 75 && h[tmp] <= 155)
                        quantified[tmp] = 6;
                    else if (h[tmp] > 155 && h[tmp] <= 210)
                        quantified[tmp] = 7;
                    else if (h[tmp] > 210 && h[tmp] <= 270)
                        quantified[tmp] = 8;
                    else if (h[tmp] > 270 && h[tmp] <= 330)
                        quantified[tmp] = 9;
                    else
                        quantified[tmp] = 10;
                }
                else if (s[tmp] > 0.1 && s[tmp] <= 0.5 && v[tmp] > 0.5 && v[tmp] <= 1.0)
                {
                    if (h[tmp] > 20 && h[tmp] <= 45)
                        quantified[tmp] = 11;
                    else if (h[tmp] > 45 && h[tmp] <= 75)
                        quantified[tmp] = 12;
                    else if (h[tmp] > 75 && h[tmp] <= 155)
                        quantified[tmp] = 13;
                    else if (h[tmp] > 155 && h[tmp] <= 210)
                        quantified[tmp] = 14;
                    else if (h[tmp] > 210 && h[tmp] <= 270)
                        quantified[tmp] = 15;
                    else if (h[tmp] > 270 && h[tmp] <= 330)
                        quantified[tmp] = 16;
                    else
                        quantified[tmp] = 17;
                }
                else if (s[tmp] > 0.5 && s[tmp] <= 1.0 && v[tmp] > 0.1 && v[tmp] <= 0.5)
                {
                    if (h[tmp] > 20 && h[tmp] <= 45)
                        quantified[tmp] = 18;
                    else if (h[tmp] > 45 && h[tmp] <= 75)
                        quantified[tmp] = 19;
                    else if (h[tmp] > 75 && h[tmp] <= 155)
                        quantified[tmp] = 20;
                    else if (h[tmp] > 155 && h[tmp] <= 210)
                        quantified[tmp] = 21;
                    else if (h[tmp] > 210 && h[tmp] <= 270)
                        quantified[tmp] = 22;
                    else if (h[tmp] > 270 && h[tmp] <= 330)
                        quantified[tmp] = 23;
                    else
                        quantified[tmp] = 24;
                }
                else if (s[tmp] > 0.5 && s[tmp] <= 1.0 && v[tmp] > 0.5 && v[tmp] <= 1.0)
                {
                    if (h[tmp] > 20 && h[tmp] <= 45)
                        quantified[tmp] = 25;
                    else if (h[tmp] > 45 && h[tmp] <= 75)
                        quantified[tmp] = 26;
                    else if (h[tmp] > 75 && h[tmp] <= 155)
                        quantified[tmp] = 27;
                    else if (h[tmp] > 155 && h[tmp] <= 210)
                        quantified[tmp] = 28;
                    else if (h[tmp] > 210 && h[tmp] <= 270)
                        quantified[tmp] = 29;
                    else if (h[tmp] > 270 && h[tmp] <= 330)
                        quantified[tmp] = 30;
                    else
                        quantified[tmp] = 31;
                }
            }    
        }
    }
}

//calc correlogram feature
void calc_correlogram_feature(uchar b[], uchar g[], uchar r[], int row, int col)
{
    //transfer BGR to HSV
    double *h = new double[row * col];
    double *s = new double[row * col];    
    double *v = new double[row * col];
	bgr2hsv(b, g, r, h, s, v, row, col);
	
	//quantify
	int *quantified = new int[row * col];
	Quantify(quantified, h, s, v, row, col);
    // for (int i = 0; i < row; i++) {
    //     for (int j = 0; j < col; j++) {
    //         printf("%3d ", quantified[i*col +j]);
    //     }
    //     printf("\n");
    // }
	
	for (int i = 0; i < row; i++)
	{
	    for (int j = 0; j < col; j++)
	    {
	        int tmp = i * col + j;
	        for (int k = 0; k < Dmax; k++)
	        {
	            if (i - k < 0 || j - k < 0 || i + k >= row || j + k >= col)
	                continue;
	            int cnttmp = 0;
	            for (int ii = i - k; ii <= i + k; ii++)
	                for (int jj = j - k; jj <= j + k; jj++)
	                    if(quantified[ii * col + jj] == quantified[tmp])
	                        cnttmp++;
	            //feature[quantified[tmp] * Dmax + k] += (double)cnttmp / (double)((2*k+1) * (2*k+1)) / (double)(row * col);
	            feature[quantified[tmp] * Dmax + k] += (double)cnttmp;
            }
	    }
	}

    for (int i = 0; i < nbins * Dmax; i++) {
        int k = 2*(i % Dmax)+1;
        int m = k*k;
        feature[i] /= (double)(m * row * col);
    }
	
	//debug output
#ifdef DEBUG
	for (int i = 0; i < 32; i++)
	{
	    for (int j = 0; j < Dmax; j++)
	        cout << feature[0][i * Dmax + j] <<"  ";
	    cout <<endl;
	}
#endif
	
	//free 
	delete [] h; h = 0;
	delete [] s; s = 0;
	delete [] v; v = 0;
	delete [] quantified; quantified = 0;
}

void print_feature()
{
    for (int i = 0; i < nbins * Dmax; i++) 
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
    char *name = "color_correlogram";

    check(argc, argv, name);

    feature = (double *)calloc(nbins * Dmax, sizeof(double));
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
	calc_correlogram_feature(b, g, r, row, col);

    //print
    print_feature();
    
    free(feature);
    return 0;
}
