#include "ImageProcess.h"
#include <vector>
using std::vector;
using namespace cv;

//High-Frequency-Emphasis Filters
Mat Butterworth_Homomorphic_Filter(Size sz, float D, float n, float high_h_v_TB, float low_h_v_TB, Mat& realIm)
{
	Mat single(sz.height, sz.width, CV_32F);
	Point centre = Point(sz.height / 2, sz.width / 2);
	float radius;
	float upper = high_h_v_TB;
	float lower = low_h_v_TB;
	float dpow = D * D;
	float W = (upper - lower);
	for (int i = 0; i < sz.height; i++) {
		for (int j = 0; j < sz.width; j++) {
			radius = pow((float)(i - centre.x), 2) + pow((float)(j - centre.y), 2);
			float r = exp(-n * radius / dpow);
			if (radius < 0)
				single.at<float>(i, j) = upper;
			else
				single.at<float>(i, j) = W * (1 - r) + lower;
		}
	}

	single.copyTo(realIm);
	Mat butterworth_complex;
	//make two channels to match complex
	Mat butterworth_channels[] = { Mat_<float>(single), Mat::zeros(sz, CV_32F) };
	merge(butterworth_channels, 2, butterworth_complex);

	return butterworth_complex;
}
Mat Butterworth_Filter(Size sz, float D, float n, float high_h_v_TB, float low_h_v_TB, Mat& realIm)
{
	Mat single(sz.height, sz.width, CV_32F);
	Point centre = Point(sz.height / 2, sz.width / 2);
	float radius;
	float upper = high_h_v_TB;
	float lower = low_h_v_TB;
	float W = (upper - lower);
	for (int i = 0; i < sz.height; i++) {
		for (int j = 0; j < sz.width; j++) {
			radius = pow((float)(i - centre.x), 2) + pow((float)(j - centre.y), 2);
			if (radius < 0)
				single.at<float>(i, j) = upper;
			else
				single.at<float>(i, j) = W * (1 - (1 / (1 + pow((radius / D), n)))) + lower;
		}
	}

	single.copyTo(realIm);
	Mat butterworth_complex;

	Mat butterworth_channels[] = { Mat_<float>(single), Mat::zeros(sz, CV_32F) };
	merge(butterworth_channels, 2, butterworth_complex);

	return butterworth_complex;
}
Mat Filter(Size sz, float D, float n, float high_h_v_TB, float low_h_v_TB, Mat& realIm)
{
	Mat single(sz.height, sz.width, CV_32F);
	Point centre = Point(sz.height / 2, sz.width / 2);
	float radius;
	float upper = high_h_v_TB;
	float lower = low_h_v_TB;
	float W = (upper - lower);
	for (int i = 0; i < sz.height; i++) {
		for (int j = 0; j < sz.width; j++) {
			radius = sqrt(pow((float)(i - centre.x), 2) + pow((float)(j - centre.y), 2));
			if (radius < 0)
				single.at<float>(i, j) = upper;
			else
				single.at<float>(i, j) = 1 / (1 + pow(radius, -n));
		}
	}

	single.copyTo(realIm);
	Mat butterworth_complex;

	Mat butterworth_channels[] = { Mat_<float>(single), Mat::zeros(sz, CV_32F) };
	merge(butterworth_channels, 2, butterworth_complex);

	return butterworth_complex;
}
//DFT 返回功率谱Power
Mat Fourier_Transform(Mat frame_bw, Mat& image_complex, Mat &image_phase, Mat &image_mag)
{
	Mat frame_log;
	frame_bw.convertTo(frame_log, CV_32F);
	frame_log = frame_log / 255;
	frame_log += 1;
	log(frame_log, frame_log); // log(1 + Mag)

	Mat padded;
	int M = getOptimalDFTSize(frame_log.rows);
	int N = getOptimalDFTSize(frame_log.cols);
	copyMakeBorder(frame_log, padded, 0, M - frame_log.rows, 0, N - frame_log.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat image_planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	merge(image_planes, 2, image_complex);


	dft(image_complex, image_complex);

	split(image_complex, image_planes);
	phase(image_planes[0], image_planes[1], image_phase);
	magnitude(image_planes[0], image_planes[1], image_mag);

	//Power
	pow(image_planes[0], 2, image_planes[0]);
	pow(image_planes[1], 2, image_planes[1]);

	Mat Power = image_planes[0] + image_planes[1];

	return Power;
}
void Inv_Fourier_Transform(Mat input, Mat& inverseTransform)
{
	Mat result;
	idft(input, result, DFT_SCALE);

	exp(result, result);

	vector<Mat> planes;
	split(result, planes);
	magnitude(planes[0], planes[1], planes[0]);
	planes[0] = planes[0] - 1.0;
	normalize(planes[0], planes[0], 0, 255, CV_MINMAX);

	planes[0].convertTo(inverseTransform, CV_8U);
}
//SHIFT
void Shifting_DFT(Mat &fImage)
{
	Mat tmp, q1, q2, q3, q4;
	//奇数图像大小取整
	fImage = fImage(Rect(0, 0, fImage.cols & -2, fImage.rows & -2));
	int cx = fImage.cols / 2;
	int cy = fImage.rows / 2;

	//获取四个象限
	q2 = fImage(Rect(0, 0, cx, cy));
	q1 = fImage(Rect(cx, 0, cx, cy));
	q3 = fImage(Rect(0, cy, cx, cy));
	q4 = fImage(Rect(cx, cy, cx, cy));

	//二四象限交换
	q2.copyTo(tmp);
	q4.copyTo(q2);
	tmp.copyTo(q4);

	//一三象限交换
	q1.copyTo(tmp);
	q3.copyTo(q1);
	tmp.copyTo(q3);
}

void homomorphicFiltering(Mat src, Mat& dst, float D0, float n, float rL, float rH)
{
	Mat img;
	Mat imgHls;
	vector<Mat> vHls;

	if (src.channels() == 3)
	{
		cvtColor(src, imgHls, CV_BGR2HSV);
		split(imgHls, vHls);
		vHls[2].copyTo(img);
	}
	else
		src.copyTo(img);

	//DFT
	//cout<<"DFT "<<endl;
	Mat img_complex, img_mag, img_phase;
	Mat fpower = Fourier_Transform(img, img_complex, img_phase, img_mag);
	Shifting_DFT(img_complex);
	//int D0 = getRadius(fpower,0.15);
	Shifting_DFT(fpower);
	Shifting_DFT(img_mag);
	int w = img_complex.cols;
	int h = img_complex.rows;
	//BHPF
	//  Mat filter,filter_complex;
	//  filter = BHPF(h,w,D0,n);
	//  Mat planes[] = {Mat_<float>(filter), Mat::zeros(filter.size(), CV_32F)};
	//  merge(planes,2,filter_complex);

	Mat filter, filter_complex;
	filter_complex = Butterworth_Homomorphic_Filter(Size(w, h), D0, n, rH, rL, filter);
	//filter_complex = Butterworth_Filter(Size(w, h), D0, n, rH, rL, filter);
	//filter_complex = Filter(Size(w, h), D0, n, rH, rL, filter);
	//dft*mask
	mulSpectrums(img_complex, filter_complex, filter_complex, 0);
	Mat s;
	Mat butterworth_channels[2];
	split(filter_complex, butterworth_channels);

	Shifting_DFT(filter_complex);
	//IDFT
	Mat result;
	Inv_Fourier_Transform(filter_complex, result);

	if (src.channels() == 3)
	{
		vHls.at(2) = result(Rect(0, 0, src.cols, src.rows));
		merge(vHls, imgHls);
		cvtColor(imgHls, dst, CV_HSV2BGR);
	}
	else {
		resize(result, result, src.size());
		result.copyTo(dst);
	}

}
