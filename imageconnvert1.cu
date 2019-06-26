#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>
using namespace cv;
using namespace std;
//nvcc -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin" -o imageconnvert1 imageconnvert1.cu -I "C:\opencv\build\include" -l "C:\opencv\build\x64\vc15\lib\opencv_world341"
//imwrite("C:\\Users\\Austin\\Pictures\\wallpapers\\IMG_3581GR.JPG", img_Grey);
// test repo from another
__global__ void kernel (unsigned char *d_in,unsigned char* d_out, int width, int height, int widthStep, int channels) {
    int x = blockIdx . x * blockDim . x + threadIdx . x ;
    int y = blockIdx . y * blockDim . y + threadIdx . y ;

    int temp;

    if (x < width && y < height) {
        int i = y;
        int j = x;
        for(int k=0; k< channels; k++) {
            temp = d_in[i*widthStep + j*channels + k];
            temp = 255-d_in[i*widthStep + j*channels + k];
            d_out[i*widthStep + j*channels + k]=temp;
        }

    }
}
void serial(int time){
	Mat input;
	input = imread("7.jpg", IMREAD_COLOR);		
	namedWindow("Input Image", WINDOW_AUTOSIZE);
	imshow("Input Image", input);
	int i,j;
	for(i=0;i<input.rows;i++){
		for(int j=0;j<input.cols;j++){
			int k =0;
			for(k=0;k<3;k++){
				//if(i==0&&j==0){printf("serial before - r- %d,g-%d,b-%d",input.at<Vec3b>(Point(j,i))[k],input.at<Vec3b>(Point(j,i))[k],input.at<Vec3b>(Point(j,i))[k]);}
				input.at<Vec3b>(Point(j,i))[k] = 255-input.at<Vec3b>(Point(j,i))[k];
			}
		}
	}	
	imshow("Output", input); 
	imwrite("C:\\Users\\U5988130\\Desktop\\finalprojectParallel\\new7.jpg", input);		
	waitKey(time);
}
void cudaCal(int time){
	IplImage* inputimage = cvLoadImage("7.jpg", CV_LOAD_IMAGE_UNCHANGED);
	IplImage* outputImage = cvCreateImage(cvGetSize(inputimage), IPL_DEPTH_8U,inputimage->nChannels);

	unsigned char *h_out = (unsigned char*)outputImage->imageData;
	unsigned char *h_in =  (unsigned char*)inputimage->imageData;

	int width     = inputimage->width;
	int height    = inputimage->height;
	int widthStep = inputimage->widthStep;
	int channels  = inputimage->nChannels;

	unsigned char* d_in;
	unsigned char* d_out;
	cudaMalloc((void**) &d_in, width*height*channels);
	cudaMalloc((void**) &d_out, width*height*channels);

	cudaMemcpy(d_in, h_in, width*height*channels, cudaMemcpyHostToDevice);
	dim3 dimBlock (32,32); // can't more than 32*32
	dim3 grid ((width+dimBlock.x-1)/dimBlock.x,(height+dimBlock.y-1)/dimBlock.y ); 
	kernel<<<grid, dimBlock>>>(d_in, d_out, width, height, widthStep, channels);

	cudaMemcpy(h_out, d_out, width * height * channels, cudaMemcpyDeviceToHost);
	cudaFree(d_in);
	cudaFree(d_out);

	cvShowImage("Original", inputimage);
	cvShowImage("CUDAChange", outputImage);

	waitKey(time);

	cvReleaseImage(&inputimage);
	cvReleaseImage(&outputImage);
        
		
}
int main(int argc, char** argv)
{
	clock_t start_t, end_t, total_t;
    printf("put 1 for serial version \n put 2 for cuda version \n put 3 for conclude performance between CUDA and CPU \n");
	printf("Enter : ");
	int val;
	scanf("%d",&val);
	if(val==1){
		serial(0);
	}
	else if(val==2){
		cudaCal(0);
	}
	else if(val == 3){
		start_t = clock();
		serial(1);
		end_t = clock();
		total_t = (double)(end_t - start_t);
		printf("Total time taken by CPU: %d\n", total_t  );

		start_t = clock();
		cudaCal(1);
		end_t = clock();
		total_t = (double)(end_t - start_t);
		printf("Total time taken by GPU: %d\n", total_t  );
	}
}

