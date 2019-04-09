#include "MSRCR.h"
#include "ImageProcess.h"
#define MAT_EXPORTS
#include "MatTools.h"
using namespace cv;

Msrcr msrcr;
static VideoCapture *video = nullptr;

MAT_API bool _stdcall readImage(const char * filename, void(*Frame)(unsigned char *image, int width, int height, int step))
{
	Mat img = imread(filename);
	if (img.empty())return false;
	Frame(img.data, img.cols, img.rows, img.step[0]);
	return true;
}

MAT_API bool _stdcall openVideo(const char * video_path)
{
	if (video != nullptr) {
		video->release();
		delete video;
		video = nullptr;
	}
	if (video == nullptr) {
		video = new VideoCapture(video_path);
		if (!video->isOpened()) {
			delete video;
			video = nullptr;
			return false;
		}
		return true;
	}
	return false;
}

MAT_API void _stdcall closeVideo()
{
	if (video != nullptr) {
		video->release();
		delete video;
		video = nullptr;
	}
}

MAT_API bool _stdcall getVideoFrame(void(*Frame)(unsigned char *image, int width, int height, int step), bool isResize, int width, int height)
{
	Mat frame;
	if (!video->read(frame)) { return false; }
	if (isResize)
		resize(frame, frame, Size(width, height));
	Frame(frame.data, frame.cols, frame.rows, frame.step[0]);
	return true;
}

MAT_API void _stdcall videoFrame(const char * video_path, void(*Frame)(unsigned char *image, int width, int height, int step, bool &stop), bool isResize, int width, int height)
{
	bool stop(false);
	cv::VideoCapture capture(video_path);
	if (!capture.isOpened())
	{
		Frame(nullptr, 0, 0, 0, stop);
		return;
	}
	double rate = capture.get(CV_CAP_PROP_FPS);
	std::cout << "Ö¡ÂÊ£º" << rate << std::endl;
	cv::Mat frame;
	int delay = 1000 / rate;
	while (!stop)
	{
		if (!capture.read(frame))
		{
			break;
		}
		if (isResize)
			resize(frame, frame, Size(width, height));
		Frame(frame.data, frame.cols, frame.rows, frame.step[0], stop);
		waitKey(delay);
		frame.release();
	}
	Frame(nullptr, 0, 0, 0, stop);
}

MAT_API void frameScreen(int width, int height, void(*Frame)(unsigned char *image, int width, int height, int step, bool &stop))
{
	bool stop(false);
	double rate = 60;
	cv::Mat frame;
	int delay = 1000 / rate;
	while (!stop)
	{
		getScreen(frame);
		if (frame.empty()) {
			break;
		}
		resize(frame, frame, Size(width, height));
		Frame(frame.data, frame.cols, frame.rows, frame.step[0], stop);
		waitKey(delay);
		frame.release();
	}
	Frame(nullptr, 0, 0, 0, stop);
}

MAT_API void catchScreen(void(*Frame)(unsigned char* image, int width, int height, int step), bool isResize, int width, int height)
{
	Mat frame;
	getScreen(frame);
	if (frame.empty()) {
		Frame(nullptr, 0, 0, 0);
	}
	if (isResize)
		resize(frame, frame, Size(width, height));
	cvtColor(frame, frame, COLOR_BGRA2BGR);
	Frame(frame.data, frame.cols, frame.rows, frame.step[0]);
	frame.release();
}


void image_enhancement(Mat & src, Mat & dst, ImageProMethodData Method)
{
	Mat image = src.clone();
	if ((Method.method&EQUALIZATION_HSV) == EQUALIZATION_HSV)
	{
		cvtColor(image, image, COLOR_BGR2HSV);
		Mat hsv[3];
		split(image, hsv);
		Ptr<CLAHE> clahe = createCLAHE(Method.equalihsvdata.clipLimit, Method.equalihsvdata.tileGridSize);
		clahe->apply(hsv[2], hsv[2]);
		merge(hsv, image.channels(), image);
		cvtColor(image, image, COLOR_HSV2BGR);
	}
	if ((Method.method&EQUALIZATION_LAB) == EQUALIZATION_LAB)
	{
		Mat bgr[3];
		cvtColor(image, image, COLOR_BGR2Lab);
		split(image, bgr);
		Ptr<CLAHE> clahe = createCLAHE(Method.equalilabdata.clipLimit, Method.equalilabdata.tileGridSize);
		for (int i = 0; i < image.channels(); i++) {
			clahe->apply(bgr[i], bgr[i]);
		}
		merge(bgr, image.channels(), image);
		cvtColor(image, image, COLOR_Lab2BGR);
	}
	if ((Method.method&HOMOMORPHIC_FILTERING) == HOMOMORPHIC_FILTERING)
	{
		cvtColor(image, image, COLOR_BGR2HSV);
		Mat hsv[3];
		split(image, hsv);
		homomorphicFiltering(hsv[2], hsv[2], Method.hfdata.D0, Method.hfdata.c, Method.hfdata.rL, Method.hfdata.rH);
		merge(hsv, image.channels(), image);
		cvtColor(image, image, COLOR_HSV2BGR);
	}
	if ((Method.method&RETINEX) == RETINEX)
		msrcr.Retinex(image, image, Method.retinexdata.sigma);
	if ((Method.method&MULTI_SCALE_RETINEX) == MULTI_SCALE_RETINEX)
		msrcr.MultiScaleRetinex(image, image, Method.multiretinexdata.weight, Method.multiretinexdata.sigema);
	if ((Method.method&MULTI_SCALE_RETINEX_CR) == MULTI_SCALE_RETINEX_CR)
		msrcr.MultiScaleRetinexCR(image, image, Method.multiretinexdata.weight, Method.multiretinexdata.sigema);
	image.copyTo(dst);
}