 #include <iostream>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;


//based on the input "image" a binary image is generated which is stored in "kimage"

void kmeans(const Mat image, Mat &kimage)
{
	//here we are creating 2 groups.So, for our algorithm k=2
	kimage.create(image.rows, image.cols, CV_8UC1);
	double mean_a = 100.0;//initialize mean for group1 
	double mean_b = 200.0;//initialize mean for group2

	double sigma_a, sigma_b, sum_a, sum_b; //store the sum of grouped pixel values
	int counter_a, counter_b; //store the number of pixels clustered into each group

	for (int i = 0; i < 20; i++)
	{
		counter_a = 0;
		counter_b = 0;
		sum_a = 0.0;
		sum_b = 0.0;
		for (int x = 0; x < image.cols; x++)
		{
			for (int y = 0; y < image.rows; y++)
			{
				double distance_a = abs((double)image.ptr<uchar>(y)[x] - mean_a);//calculate the Euclidean distance from current pixel value to mean_a
				double distance_b = abs((double)image.ptr<uchar>(y)[x] - mean_b);//calculate the Euclidean distance from current pixel value to mean_b

				//assign the current pixel to group 1 or group 2

				if (distance_a > distance_b)//if the pixel is far away from mean_a then 
				{
					//pixel is grouped to group 2

					sum_b += image.ptr<uchar>(y)[x];
					counter_b++;
					kimage.ptr<uchar>(y)[x] = 255;
				}

				else
				{
					//pixel is grouped to group 1

					sum_a += image.ptr<uchar>(y)[x];
					counter_a++;
					kimage.ptr<uchar>(y)[x] = 0;
				}

			}
		}
		mean_a = sum_a / counter_a; //after every iteration it updates new mean for group 1
		mean_b = sum_b / counter_b; //after every iteration it updates new mean for group 2
		sigma_a = sqrt(((pow(sum_a, 2)) / counter_a) - pow(mean_a, 2));
		sigma_b = sqrt(((pow(sum_b, 2)) / counter_b) - pow(mean_b, 2));

	}

	imshow("kmeans", kimage);
	cout << mean_a << endl;
	cout << mean_b << endl << counter_a << endl << counter_b << endl << sigma_a << endl << sigma_b;
}


//using gaussian distribution assign the weight 
double computeGauss(double x, double mean, double sigma)
{
	double p_x = (1 / (sqrt(2 * 3.14*pow(sigma, 2.0)))) * exp(-0.5 * pow((x - mean) / sigma, 2.0));
	return p_x;
}

void em_algo(const Mat image, Mat &label_img)
{
	int counter_a, counter_b;

	label_img.create(image.rows, image.cols, CV_8UC1);
	//initialize standard deviation value 
	double sigma_a1 = 220;
	double sigma_b1 = 300;

	//get mean value based on kmeans 
	double mean_a1 = 113.0;
	double mean_b1 = 185.0;

	//get value based on kmeans
	//p_a=counter_a/counter_a+counter_b
	double p_a = 0.1873;
	double p_b = 1 - p_a;

	double n_a, n_b;
	double sigma_a0, sigma_b0;
	double sum_a, sum_b;

	n_a = 0;
	n_b = 0;

	sigma_a0 = 0;
	sigma_b0 = 0;
	double sum = 0;
	sum_a = 0;
	sum_b = 0;


	counter_a = 51347;
	counter_b = 222749;

	for (int i = 0; i < 20; i++)
	{

		for (int r = 0; r < image.cols ; r++)
		{
			for (int c = 0; c < image.rows; c++)
			{
				double x = (double)image.ptr<uchar>(r)[c];
				//gaussian values per each pixel belonging to group

				double p_xa = computeGauss(x, mean_a1, sigma_a1);
				double p_xb = computeGauss(x, mean_b1, sigma_b1);
				//a=p(a) * p(x|a) / p(a) * p(x|a) + p(b) * p(x|b)
				//b=p(b) * p(x|b) / p(a) * p(x|a) + p(b) * p(x|b)

				double a = p_a * p_xa / (p_a*p_xa + p_b * p_xb);
				double b = p_b * p_xb / (p_a*p_xa + p_b * p_xb);

				if (a > b)
				{
					label_img.ptr<uchar>(r)[c] = 0;

				
				}
				else
				{
					label_img.ptr<uchar>(r)[c] = 255;

				
				}

				sum_a += a * x;
				sum_b += b * x;

				n_a += a;
				n_b += b;

				sigma_a0 += a * (pow((x - mean_a1), 2));
				sigma_b0 += b * (pow((x - mean_b1), 2));

			}
		}


		cout << "p_a" << p_a << endl << "p_b" << p_b << endl;

		//mean_a = sum(a * x) / sum(a)
		//mean_b = sum(b * x) / sum(b)
		mean_a1 = sum_a / n_a;
		mean_b1 = sum_b / n_b;
		cout << "mean_a1" << mean_a1 << endl << "mean_b1" << mean_b1;

		//standard deviation 
		//sigma_a ^ 2 = sum(a * ((x - mean_a) ^ 2)) / sum(a)
		//sigma_b ^ 2 = sum(b * ((x - mean_b) ^ 2)) / sum(b)
		sigma_a1 = sqrt((sigma_a0) / (n_a));
		sigma_b1 = sqrt((sigma_b0) / (n_b));
		cout << "sigma_a1" << sigma_a1 << endl << "sigma_b1" << sigma_b1;
	}
	imshow("em_image", label_img);

}

int main(int argc, char** argv)
{
	//load the image
	Mat image = imread("Original_Image.png");
	resize(image, image, Size(image.cols / 2, image.rows / 2));

	//Convert the image to grayscale
	cvtColor(image, image, CV_BGR2GRAY);
	char ch = waitKey(1);

	Mat kimage;
	Mat label_img;
	kmeans(image, kimage);
	em_algo(kimage, label_img);
	while (1)
	{

		Mat canny_img;
		//Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);
		Canny(kimage, canny_img, 50, 150, 3);

		canny_img.convertTo(canny_img, CV_8U);

		//display canny image
		imshow("canny_image", canny_img);



		// Blur the image to reduce noise 
		Mat img_blur;

		medianBlur(kimage, img_blur, 5);

		// Create a vector for detected circles
		vector<Vec3f> circles;

		// Apply Hough Transform
		HoughCircles(img_blur, circles, CV_HOUGH_GRADIENT, 1, kimage.rows / 8, 200, 10, 5, 50);

		// Draw detected circles
		for (size_t i = 0; i < circles.size(); i++)
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int r = cvRound(circles[i][2]);
			circle(kimage, center, 3, Scalar(255, 0, 0), -1, 8, 0);
			circle(kimage, center, r, Scalar(0, 255, 255), 3, 8, 0);
		}

		//Display the detected circle(s)
		namedWindow("hough_image", CV_WINDOW_AUTOSIZE);
		imshow("hough_image", kimage);

		//Wait for the user to exit the program
		if (ch == 27)
			break;
	}

	return 1;

}