#ifndef __MATTOOLS_H__
#define __MATTOOLS_H__

#ifdef MAT_EXPORTS
#define MAT_API __declspec(dllexport)
#else
#define MAT_API __declspec(dllimport)
#endif

#include <opencv2\opencv.hpp>
#include <vector>
using std::vector;
using cv::Mat;
using cv::Size;

#define EQUALIZATION_HSV		1
#define EQUALIZATION_LAB		2
#define HOMOMORPHIC_FILTERING	4
#define RETINEX					8
#define MULTI_SCALE_RETINEX		16
#define MULTI_SCALE_RETINEX_CR	32

struct HFData
{
	float D0;
	float c;
	float rL;
	float rH;
	void Default()
	{
		D0 = 100;
		c = 1;
		rL = 0.5;
		rH = 2;
	}
};
struct EqualiHSVData
{
	double clipLimit;
	Size tileGridSize;
	void Default()
	{
		clipLimit = 2.0;
		tileGridSize = Size(8, 8);
	}
};
struct EqualiLABData
{
	double clipLimit;
	Size tileGridSize;
	void Default()
	{
		clipLimit = 2.0;
		tileGridSize = Size(8, 8);
	}
};
struct RetinexData
{
	double sigma;
	void Default()
	{
		sigma = 100;
	}
};
struct MultiRetinexData
{
	vector<double> sigema;
	vector<double> weight;
	void Default()
	{
		vector<double>().swap(sigema);
		vector<double>().swap(weight);
		for (int i = 0; i < 3; i++)
			weight.push_back(1. / 3);
		sigema.push_back(30);
		sigema.push_back(100);
		sigema.push_back(150);
	}
};
class ImageProMethodData
{
public:
	int method;
	struct HFData hfdata;
	struct EqualiHSVData equalihsvdata;
	struct EqualiLABData equalilabdata;
	struct RetinexData retinexdata;
	struct MultiRetinexData multiretinexdata;
	ImageProMethodData() {}
	ImageProMethodData(int Method)
	{
		method = Method;
		Default();
	}
	void Default() {
		if ((method&EQUALIZATION_HSV) == EQUALIZATION_HSV)equalihsvdata.Default();
		if ((method&EQUALIZATION_LAB) == EQUALIZATION_LAB)equalilabdata.Default();
		if ((method&HOMOMORPHIC_FILTERING) == HOMOMORPHIC_FILTERING)hfdata.Default();
		if ((method&RETINEX) == RETINEX)retinexdata.Default();
		if ((method&MULTI_SCALE_RETINEX) == MULTI_SCALE_RETINEX)multiretinexdata.Default();
		if ((method&MULTI_SCALE_RETINEX_CR) == MULTI_SCALE_RETINEX_CR)multiretinexdata.Default();
	}
};

extern "C" MAT_API bool _stdcall readImage(const char* filename, void(*Frame)(unsigned char* image, int width, int height, int step));
extern "C" MAT_API bool _stdcall openVideo(const char * video_path);
extern "C" MAT_API void _stdcall closeVideo();
extern "C" MAT_API bool _stdcall getVideoFrame(void(*Frame)(unsigned char *image, int width, int height, int step), bool isResize, int width, int height);
extern "C" MAT_API void _stdcall videoFrame(const char * video_path, void(*Frame)(unsigned char *image, int width, int height, int step, bool &stop), bool isResize, int width, int height);
extern "C" MAT_API void _stdcall frameScreen(int width, int height, void(*Frame)(unsigned char* image, int width, int height, int step, bool &stop));
extern "C" MAT_API void _stdcall catchScreen(void(*Frame)(unsigned char* image, int width, int height, int step), bool isResize, int width, int height);

MAT_API void getScreen(Mat & frame);
MAT_API void image_enhancement(Mat &src, Mat&dst, ImageProMethodData Method);
MAT_API void CreatMat(Mat &dst, unsigned char* src, int cols, int rows, int srider, int channel);
MAT_API void facialPoseCorrection(const Mat &src, Mat &dst, int left_eye_x, int left_eye_y, int right_eye_x, int right_eye_y);


#endif // __MATTOOLS_H__
