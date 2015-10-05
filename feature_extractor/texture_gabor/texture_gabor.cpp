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
#define nScale 5
#define nDirection 6
#define ntype 2
#define width 2
typedef unsigned int uint;

//feature
double feature[nScale * nDirection * ntype];

//Gabor filter
double gaborReal[nScale][nDirection][2*width+1][2*width+1];
double gaborImag[nScale][nDirection][2*width+1][2*width+1];

//generate Gabor filter
//M:number of scales
//N:number of directions
//W:number of filter sizes
void generateGabor()
{
    int M = nScale;
    int N = nDirection;
    double Uh = 0.4;
    double Ul = 0.05;
    double a = pow( (Uh/Ul) , -1.0/(M-1) );
       
    double W = 0.5;
        
    double xx[2*width+1][2*width+1], yy[2*width+1][2*width+1];
    
    
    for (int m = 0; m < M; m++)
    {
        double sigma_x = (a+1) * sqrt(2.0*log(2.0)) / (2.0 * PI * pow(a, m) * (a-1) * Ul);
        double sigma_y = 1.0 / ( 2*PI * tan(PI/2*N) * sqrt( abs((Uh*Uh)/2.0*log(2.0) - (1.0/2.0*PI*sigma_x)*(1.0/2.0*PI*sigma_x) )) );
        for (int n = 0; n < N; n++)
        {
            double theta = n * PI / N;
            double costheta = cos(theta);
            double sintheta = sin(theta);
            double a1 = pow(a, -m);//加速
                    
            for (int i = 0; i < 2*width + 1; i++)
            {
                for (int j = 0; j < 2*width+1; j++)
                {
                    xx[i][j] = a1 * ( (double)(j-width)*costheta + (double)(i-width)*sintheta );
                    yy[i][j] = a1 * ( (double)(j-width)*(-sintheta) + (double)(i-width)*costheta );
                }
            }
            
            double a2 = a1 / (2 * PI * sigma_x * sigma_y);
            for (int i = 0; i < 2*width + 1; i++)
            {
                for (int j = 0; j < 2*width+1; j++)
                {
                    double b = exp( -0.5*( xx[i][j]*xx[i][j]/(sigma_x*sigma_x) - yy[i][j]*yy[i][j]/(sigma_y*sigma_y) ) );
                    double tmpreal = cos(2.0*PI*W*xx[i][j]);
                    double tmpimag = sin(2.0*PI*W*xx[i][j]);
                    gaborReal[m][n][i][j] = a2 * b * tmpimag;
                    gaborImag[m][n][i][j] = a2 * b * tmpreal;
                }
            }
    
        }
    }
}

//calc feature_gabor
void calcGaborFeature(Mat src, int m, int n, int row, int col, double feas[])
{
    Point anchor;
    double delta = 0.0;
    int ddepth = -1;
    anchor = Point(-1, -1);
    Mat kerR(2*width+1, 2*width+1, CV_64F, gaborReal[m][n][0]); 
    Mat kerI(2*width+1, 2*width+1, CV_64F, gaborImag[m][n][0]);
    
    Mat R, I;
    filter2D(src, R, ddepth , kerR, anchor, delta, BORDER_DEFAULT );
    filter2D(src, I, ddepth , kerI, anchor, delta, BORDER_DEFAULT );
    
    double miu = 0.0;
    for (int i = 0; i < row; i++)
    {
        double *pR = R.ptr<double>(i);
        double *pI = I.ptr<double>(i);
        for (int j = 0; j < col; j++)
        {
            miu += sqrt(pR[j]*pR[j] + pI[j]*pI[j]);
            //cout << pR[j] << " " << pI[j] << endl;
        }
    }
    miu /= (double)(row * col);
    
    double sigma = 0.0;
    for (int i = 0; i < row; i++)
    {
        double *pR = R.ptr<double>(i);
        double *pI = I.ptr<double>(i);
        for (int j = 0; j < col; j++)
        {
            double tmp = sqrt(pR[j]*pR[j] + pI[j]*pI[j]) - miu;
            sigma += tmp * tmp;
        }
    }
    sigma /= (double)(row * col);
    
    feas[0] = miu;
    feas[1] = sigma;
}

//calc feature gabor texture
void calc_feature_texture_gabor(uchar gray[], int row, int col)
{
    int size = row * col;
    double *g = new double[row*col];
    for (int i = 0; i < size; i++)
        g[i] = (double)gray[i];
    Mat src(row, col, CV_64F, g);

    double feas[2];    

    for (int m = 0; m < nScale; m++)
    {
        for (int n = 0; n < nDirection; n++)
        {
            calcGaborFeature(src, m, n, row, col, feas);
            int offset = m * nDirection + n;
            feature[offset * ntype] = feas[0];
            feature[offset * ntype + 1] = feas[1];
        }
    }
    
    //free
    delete []g;
}

void print_feature()
{
    for (int i = 0; i < nScale * nDirection * ntype; i++) 
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
    char *name = "texture_gabor";

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

	memset(feature, 0, nScale * nDirection * sizeof(double));
	generateGabor();
	
    uchar *gray = image.ptr<uchar>(0);	

	int row = image.rows;
	int col = image.cols;
		
	calc_feature_texture_gabor(gray, row, col);

    //print
    print_feature();

	return 0;
}
