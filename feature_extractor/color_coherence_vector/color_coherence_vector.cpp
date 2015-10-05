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

const int nbins = 32;
const int ntype = 2;
typedef unsigned int uint;

//feature
int *feature;

//calc bgr2hsv
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
            if (maxtmp == r[tmp])
            {
                h[tmp] = (g[tmp] - b[tmp]) / (maxtmp - mintmp) * 60.0;
            }
            if (maxtmp == g[tmp])
            {
                h[tmp] = 120.0 + (b[tmp] - r[tmp]) / (maxtmp - mintmp) * 60.0;
            }
            if (maxtmp == b[tmp])
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

//neigbors
int go[8][2] = 
{
{1, -1},
{1, 0},
{1, 1},
{-1, -1},
{-1, 0},
{-1, 1},
{0, 1},
{0, -1},
};

//calc color coherence vector
void calc_color_coherence_vector_feature(uchar b[], uchar g[], uchar r[], int row, int col)
{
    //transfer to HSV
    double *h = new double[row * col];
    double *s = new double[row * col];    
    double *v = new double[row * col];
	bgr2hsv(b, g, r, h, s, v, row, col);
	
	//quantify
	int *quantified = new int[row * col];
	Quantify(quantified, h, s, v, row, col);
	
	//set mark matrix and init it
	bool *mark = new bool[row * col];
	memset(mark, 0, row * col * sizeof(bool));
	
	//set threshold
	double threshold = (double)(row * col) * 0.01;
	
	//initial coherence vector
	int alpha[nbins], beta[nbins];
	memset(alpha, 0, nbins * sizeof(int));
	memset(beta, 0, nbins * sizeof(int));
		
	//set queue for BFS
	queue<int> Q;
	
	//calc coherence vector
	for (int i = 0; i < row; i++)
	{
	    for (int j = 0; j < col; j++)
	    {
	        while(!Q.empty())
	            Q.pop();
	        int cnt = 0;
	        int p = quantified[i * col + j];
	        if (mark[i * col + j] == true)
	            continue;
	        Q.push(i * col + j);
	        cnt++;
	        while (Q.empty() == false)
	        {
	            int q = Q.front();
	            Q.pop();
	            int ii = q / col;
	            int jj = q % col;
	            for (int k = 0; k < 8; k++)
	            {
	                int ik = go[k][0], jk = go[k][1];
	                if (ii + ik < 0 || ii + ik >= row || jj + jk < 0 || jj + jk >= col)
	                    continue;
	                if ( mark[(ii+ik)*col + (jj+jk)] == true)
	                    continue;
	                if (quantified[(ii+ik)*col + (jj+jk)] != p)
	                    continue;
	                mark[(ii+ik)*col + (jj+jk)] = true;
	                Q.push((ii+ik)*col + (jj+jk));
	                cnt++;
	            }
	        }
	        if (cnt >= threshold)
	            alpha[p]++;
	        else
	            beta[p]++;
	    }
	}
	
	//threshold
	for (int i = 0; i < nbins; i++)
	{
	    feature[i * ntype] = alpha[i];
	    feature[i * ntype + 1] = beta[i];
	}
	
	//free
	delete [] h; h = 0;
	delete [] s; s = 0;
	delete [] v; v = 0;
	delete [] quantified; quantified = 0;
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

int main(int argc, char** argv)
{
    char *name = "color_coherence_vector";

    check(argc, argv, name);

    feature = (int *)calloc(nbins * ntype, sizeof(int));
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

	//split to BGR matrix
	vector<Mat> bgr_planes;
	bgr_planes.clear();
	split( image, bgr_planes);

	uchar *b = bgr_planes[0].ptr<uchar>(0);	
	uchar *g = bgr_planes[1].ptr<uchar>(0);	
	uchar *r = bgr_planes[2].ptr<uchar>(0);	

	int row = bgr_planes[0].rows;
	int col = bgr_planes[0].cols;

	calc_color_coherence_vector_feature(b, g, r, row, col);

	//print feature
	print_feature();

	free(feature);
	return 0;
}
