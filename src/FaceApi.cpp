#include <iostream>

#ifdef _WIN32
#pragma once
#endif

#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#define FACE_EXPORTS
#include "FaceApi.h"

#include "MSRCR.h"
#include "ImageProcess.h"
#define MAT_EXPORTS
#include "MatTools.h"

#include "face_detection.h"

#include "face_alignment.h"
#include "face_identification.h"
#include "recognizer.h"
#include "math_functions.h"

using std::string;
using seeta::FaceIdentification;
using seeta::FaceAlignment;
using seeta::ImageData;
using seeta::FaceInfo;
using mtcnn::FaceDet;

using namespace cv;

static unsigned char * imagedata = nullptr;
static FaceAlignment *face_alignment = nullptr;
static FaceIdentification *face_recognizer = nullptr;
static FaceDet *FD = nullptr;
static float *feature_vec = nullptr;
static int *lankmark = nullptr;
const int fea_len = 2048;

void face_recognition(const char* img1, const char* img2);
double face_recognition_(const char* img1, const char* img2);
void Feature(const Mat &gallery_img_color, cv::Rect rect, float fea[2048]);
double recoginizer(Mat &img1, Mat &img2, vector<double> &point1, vector<double> &point2);
void recognition(Mat &img, vector<double> &point, float feature[2048]);
Mat detect(vector<double> &point, const char *imgpath, double minarea, bool re_size);
void detect(const Mat &src, vector<double> &point, double minarea, bool re_size);
void detect_s(const Mat &src, vector<vector<double>> &point, double minarea, bool re_size);

bool _stdcall create_load_models(const char* model_path)
{	
	if (FD == nullptr) {
		FD = new FaceDet();
		FD->creat_mtcnn(model_path);
	}
	else {
		FD->creat_mtcnn(model_path);
	}
	string str = model_path;
	if (face_recognizer == nullptr) {
		
		face_recognizer = new FaceIdentification();
		face_recognizer->LoadModel((str + "/seeta_fr_v1.0.bin").c_str());
	}
	else {
		face_recognizer->LoadModel((str + "/seeta_fr_v1.0.bin").c_str());
	}
	if (face_alignment == nullptr) {

		face_alignment = new FaceAlignment((str + "/seeta_fa_v1.1.bin").c_str());
	}
	if (FD == nullptr || face_recognizer == nullptr || face_alignment == nullptr)
		return false;
	return true;
}

void _stdcall del_model()throw(...)
{
	if (face_recognizer != nullptr) {
		delete face_recognizer;
		face_recognizer = nullptr;
	}
	if (FD != nullptr) {
		delete FD;
		FD = nullptr;
	}
	if (face_alignment != nullptr) {
		delete face_alignment;
		face_alignment = nullptr;
	}
	if (feature_vec != nullptr) {
		delete feature_vec;
		feature_vec = nullptr;
	}
	if (imagedata != nullptr) {
		delete imagedata;
		imagedata = nullptr;
	}
	if (lankmark != nullptr) {
		delete lankmark;
		lankmark = nullptr;
	}
}

void face_recognition(const char* img1, const char* img2)
{
	double result;
	try {
		result = face_recognition_(img1, img2);
	}
	catch (...) {
		result = -7;
	}
	ofstream out("data.txt");
	if (out.is_open()) {
		out << result << endl;
		out.close();
	}
}

double _stdcall FaceRecognition(const char * img1, const char * img2)
{
	static double result = 0;
	try {
		result = face_recognition_(img1, img2);
	}
	catch (...) {
		result = -7;
	}
	return result;
}

double _stdcall Recoginizer(
	unsigned char * CameraForm, 
	int FormWidth, int FormHeight, int FormStep, 
	unsigned char * IDFace,
	int FaceWidth, int FaceHeight, int FaceStep)
{
	Mat form;
	Mat idface;
	CreatMat(form, CameraForm, FormWidth, FormHeight, FormStep, 3);
	CreatMat(idface, IDFace, FaceWidth, FaceHeight, FaceStep, 3);
	if (FD == nullptr || face_recognizer == nullptr) {
		fprintf(stderr, "no find model!\n");
		return -2;
	}
	Mat image_1 = form;
	if (image_1.empty()) {
		fprintf(stderr, "load image fail!\n");
		return -3;
	}
	Mat dst;
	cvtColor(image_1, dst, COLOR_BGR2RGB);
	if (!FD->run(dst.data, dst.cols, dst.rows))
	{
		if (FD->face_num() == 0)return -1;
		return -4;
	}
	image_enhancement(image_1, image_1, ImageProMethodData(EQUALIZATION_HSV));
	ImageData gallery_img_data_color(image_1.cols, image_1.rows, image_1.channels());
	gallery_img_data_color.data = image_1.data;
	vector<double> point = FD->finish();
	int pp1[4] = { int(point[5 * 2]),int(point[5 * 2 + 1]),int(point[5 * 2 + 2]),int(point[5 * 2 + 3]) };
	seeta::FacialLandmark gallery_points[5];
	for (int index = 0; index < 5; ++index) {
		gallery_points[index].x = point[index * 2];
		gallery_points[index].y = point[index * 2 + 1];
	}
	Mat image_2 = idface;
	if (image_2.empty()) {
		fprintf(stderr, "load image fail!\n");
		return -5;
	}
	cvtColor(image_2, dst, COLOR_BGR2RGB);
	if (!FD->run(dst.data, dst.cols, dst.rows))
	{
		if (FD->face_num() == 0)return -1;
		return -6;
	}
	image_enhancement(image_2, image_2, ImageProMethodData(EQUALIZATION_LAB));
	ImageData probe_img_data_color(image_2.cols, image_2.rows, image_2.channels());
	probe_img_data_color.data = image_2.data;
	point = FD->finish();
	int pp2[4] = { int(point[5 * 2]),int(point[5 * 2 + 1]),int(point[5 * 2 + 2]),int(point[5 * 2 + 3]) };
	seeta::FacialLandmark probe_points[5];
	for (int index = 0; index < 5; ++index) {
		probe_points[index].x = point[index * 2];
		probe_points[index].y = point[index * 2 + 1];
	}
	float gallery_fea[2048];
	float probe_fea[2048];
	face_recognizer->ExtractFeatureWithCrop(gallery_img_data_color, gallery_points, gallery_fea);
	face_recognizer->ExtractFeatureWithCrop(probe_img_data_color, probe_points, probe_fea);

	// Caculate similarity of two faces
	float sim = face_recognizer->CalcSimilarity(gallery_fea, probe_fea);

	return double(sim);
}

FACE_API bool drawFace(unsigned char * image, int width, int height, int step, int channels, int r, int g, int b, int minarea)
{
	if (image == nullptr)return false;
	if (FD == nullptr)return false;
	Mat src;
	CreatMat(src, image, width, height, step, channels);
	if (src.empty())return false;
	vector<double> point;
	detect(src, point, minarea, false);
	if (point.empty())return false;
	Rect rect((int)point[10], (int)point[11], int(point[12] - point[10]), int(point[13] - point[11]));
	rectangle(src, Rect(rect), Scalar(r, g, b), 1, 8);
	return true;
}

FACE_API int drawFaces(unsigned char * image, int width, int height, int step, int channels, int minarea, bool drawmark, bool randColor, int r, int g, int b)
{
	if (image == nullptr)return -1;
	if (FD == nullptr)return -2;
	Mat src;
	CreatMat(src, image, width, height, step, channels);
	if (src.empty())return -3;
	vector<vector<double>> points;
	detect_s(src, points, minarea, false);
	if (points.empty())return -4;
	Mat img(height, width, CV_8UC3, image, step);
	for (size_t idx = 0; idx < points.size(); ++idx) {
		Rect rect((int)points[idx][10], (int)points[idx][11], int(points[idx][12] - points[idx][10]), int(points[idx][13] - points[idx][11]));
		if (randColor) { 
			if (drawmark) {
				for (int index = 0; index < 5; ++index) {
					circle(img, Point(points[idx][index * 2], points[idx][index * 2 + 1]), 2, Scalar(rand() % 256, rand() % 256, rand() % 256), -1);
				}
			}
			rectangle(img, rect, Scalar(rand() % 256, rand() % 256, rand() % 256), 1, 8); 
		}
		else {
			if (drawmark) {
				for (int index = 0; index < 5; ++index) {
					circle(img, Point(points[idx][index * 2], points[idx][index * 2 + 1]), 2, Scalar(b, g, r), -1);
				}
			}
			rectangle(img, rect, Scalar(r, g, b), 1, 8); 
		}
	}
	memcpy(image, img.data, sizeof(uchar)*img.cols*img.rows*img.channels());
	return 0;
}

FACE_API int* findFace(unsigned char * image, int width, int height, int step, int channels, int & xLeft, int & yLeft, int & rectWitdh, int & rectHeight, int minarea)
{
	if (image == nullptr)return nullptr;
	if (FD == nullptr)return nullptr;
	Mat src;
	CreatMat(src, image, width, height, step, channels);
	if (src.empty())return nullptr;
	vector<double> point;
	detect(src, point, minarea, false);
	if (point.empty())return nullptr;
	if (lankmark != nullptr) {
		delete[] lankmark;
		lankmark = nullptr;
	}
	if (lankmark == nullptr)
		lankmark = new int[10];
	for (int index = 0; index < 10; ++index) {
		lankmark[index] = (int)point[index];
	}
	xLeft = (int)point[10];
	yLeft = (int)point[11];
	rectWitdh = (int)(point[12] - xLeft);
	rectHeight = (int)(point[13] - yLeft);
	return lankmark;
}

FACE_API int * findFace_(unsigned char * image, int width, int height, int step, int channels, int & xLeft, int & yLeft, int & xRight, int & yRight, int minarea)
{
	if (image == nullptr)return nullptr;
	if (FD == nullptr)return nullptr;
	Mat src;
	CreatMat(src, image, width, height, step, channels);
	if (src.empty())return nullptr;
	vector<double> point;
	detect(src, point, minarea, false);
	if (point.empty())return nullptr;
	if (lankmark != nullptr) {
		delete[] lankmark;
		lankmark = nullptr;
	}
	if (lankmark == nullptr)
		lankmark = new int[10];
	for (int index = 0; index < 10; ++index) {
		lankmark[index] = (int)point[index];
	}
	xLeft = (int)point[10];
	yLeft = (int)point[11];
	xRight = (int)(point[12]);
	yRight = (int)(point[13]);
	return lankmark;
}

FACE_API bool findFaceP(unsigned char * image, int width, int height, int step, int channels, int & xLeft, int & yLeft, int & rectWitdh, int & rectHeight, int * landMark, int minarea)
{
	if (image == nullptr)return false;
	if (FD == nullptr)return false;
	Mat src;
	CreatMat(src, image, width, height, step, channels);
	if (src.empty())return false;
	vector<double> point;
	detect(src, point, minarea, false);
	if (point.empty())return false;
	xLeft = (int)point[10];
	yLeft = (int)point[11];
	rectWitdh = (int)(point[12] - xLeft);
	rectHeight = (int)(point[13] - yLeft);
	if (landMark != nullptr) {
		for (int index = 0; index < 10; ++index) {
			landMark[index] = (int)point[index];
		}
	}
	return true;
}

FACE_API bool findFaceP_(unsigned char * image, int width, int height, int step, int channels, int & xLeft, int & yLeft, int & xRight, int & yRight, int * landMark, int minarea)
{
	if (image == nullptr)return false;
	if (FD == nullptr)return false;
	Mat src;
	CreatMat(src, image, width, height, step, channels);
	if (src.empty())return false;
	vector<double> point;
	detect(src, point, minarea, false);
	if (point.empty())return false;
	xLeft = (int)point[10];
	yLeft = (int)point[11];
	xRight = (int)(point[12]);
	yRight = (int)(point[13]); 
	if (landMark != nullptr) {
		for (int index = 0; index < 10; ++index) {
			landMark[index] = (int)point[index];
		}
	}
	return true;
}

FACE_API bool _stdcall findFaceRect(unsigned char * image, int width, int height, int step, int channels, int & xLeft, int & yLeft, int & rectWitdh, int & rectHeight, int minarea)
{
	if (image == nullptr)return false;
	if (FD == nullptr)return false;
	Mat src;
	CreatMat(src, image, width, height, step, channels);
	if (src.empty())return false;
	vector<double> point;
	detect(src, point, minarea, false);
	if (point.empty())return false;
	xLeft = (int)point[10];
	yLeft = (int)point[11];
	rectWitdh = (int)(point[12] - xLeft);
	rectHeight = (int)(point[13] - yLeft);
	return true;
}

FACE_API bool _stdcall findFacePoint(unsigned char * image, int width, int height, int step, int channels, int & xLeft, int & yLeft, int & xRight, int & yRight, int minarea)
{
	if (image == nullptr)return false;
	if (FD == nullptr)return false;
	Mat src;
	CreatMat(src, image, width, height, step, channels);
	if (src.empty())return false;
	vector<double> point;
	detect(src, point, minarea, false);
	if (point.empty())return false;
	xLeft = (int)point[10];
	yLeft = (int)point[11];
	xRight = (int)(point[12]);
	yRight = (int)(point[13]);
	return true;
}

FACE_API int landMarkLen()
{
	return int(10);
}

FACE_API int featureVecLen()
{
	return int(fea_len);
}

FACE_API unsigned int featureVecLength()
{
	return unsigned int(fea_len);
}

FACE_API bool featureVec(unsigned char * image, int width, int height, int step, int channels, float * fea)
{
	if (image == nullptr)return false;
	if (fea == nullptr)return false;
	if (face_recognizer == nullptr || FD == nullptr)return false;
	Mat src;
	CreatMat(src, image, width, height, step, channels);
	if (src.empty())return false;
	ImageData img_data_color(src.cols, src.rows, src.channels());
	img_data_color.data = src.data;
	vector<double> point;
	detect(src, point, 0, false);
	if (point.empty())return false;
	seeta::FacialLandmark gallery_points[5];
	for (int index = 0; index < 5; ++index) {
		gallery_points[index].x = point[index * 2];
		gallery_points[index].y = point[index * 2 + 1];
	}
	face_recognizer->ExtractFeatureWithCrop(img_data_color, gallery_points, fea);
	return true;
}

FACE_API bool _stdcall featureVecPoint(unsigned char * image, int width, int height, int step, int channels, float * fea, int xLeft, int yLeft, int xRight, int yRight)
{
	if (image == nullptr)return false;
	if (fea == nullptr)return false;
	if (face_recognizer == nullptr || face_alignment == nullptr)return false;
	Mat src;
	CreatMat(src, image, width, height, step, channels);
	if (src.empty())return false;
	Feature(src, Rect(Point2i(xLeft, yLeft), Point2i(xRight, yRight)), fea);
	return true;
}

FACE_API bool _stdcall featureVecRect(unsigned char * image, int width, int height, int step, int channels, float * fea, int xLeft, int yLeft, int rectWitdh, int rectHeight)
{
	if (image == nullptr)return false;
	if (fea == nullptr)return false;
	if (face_recognizer == nullptr || face_alignment == nullptr)return false;
	Mat src;
	CreatMat(src, image, width, height, step, channels);
	if (src.empty())return false;
	Feature(src, Rect(xLeft, yLeft, rectWitdh, rectHeight), fea);
	return true;
}

FACE_API bool featureVecP(unsigned char * image, int width, int height, int step, int channels, int * landMark, float * fea)
{
	if (image == nullptr)return false;
	if (landMark == nullptr)return false;
	if (fea == nullptr)return false;
	if (face_recognizer == nullptr || face_alignment == nullptr)return false;
	Mat src;
	CreatMat(src, image, width, height, step, channels);
	if (src.empty())return false; 
	ImageData img_data_color(src.cols, src.rows, src.channels());
	img_data_color.data = src.data;
	seeta::FacialLandmark gallery_points[5];
	for (int index = 0; index < 5; ++index) {
		gallery_points[index].x = landMark[index * 2];
		gallery_points[index].y = landMark[index * 2 + 1];
	}
	face_recognizer->ExtractFeatureWithCrop(img_data_color, gallery_points, fea);
	return true;
}

FACE_API float* featureVecs(
	unsigned char* image, int &len,
	int width, int height, int step, int channels)
{
	if (image == nullptr)return nullptr;
	if (face_recognizer == nullptr || FD == nullptr)return nullptr;
	Mat src;
	CreatMat(src, image, width, height, step, channels);
	if (src.empty())return nullptr;
	len = 0;
	ImageData img_data_color(src.cols, src.rows, src.channels());
	img_data_color.data = src.data;
	vector<vector<double>> points;
	detect_s(src, points, 0, false);
	len = (int)points.size();
	if (len == 0)return nullptr;
	if (feature_vec != nullptr) {
		delete[] feature_vec;
		feature_vec = nullptr;
	}
	if (feature_vec == nullptr)
		feature_vec = new float[len*fea_len];
	for (int idx = 0; idx < len; ++idx) {
		seeta::FacialLandmark gallery_points[5];
		for (int index = 0; index < 5; ++index) {
			gallery_points[index].x = points[idx][index * 2];
			gallery_points[index].y = points[idx][index * 2 + 1];
		}
		face_recognizer->ExtractFeatureWithCrop(img_data_color, gallery_points, feature_vec + idx * fea_len);
	}
	return feature_vec;
}

FACE_API float * _featureVec(unsigned char * image, int width, int height, int step, int channels, int * landMark)
{
	if (image == nullptr)return nullptr;
	if (landMark == nullptr)return nullptr;
	if (face_recognizer == nullptr)return nullptr;
	Mat src;
	CreatMat(src, image, width, height, step, channels);
	if (src.empty())return nullptr;
	ImageData img_data_color(src.cols, src.rows, src.channels());
	img_data_color.data = src.data;

	if (feature_vec != nullptr) {
		delete[] feature_vec;
		feature_vec = nullptr;
	}
	if (feature_vec == nullptr)
		feature_vec = new float[fea_len];
	seeta::FacialLandmark gallery_points[5];
	for (int index = 0; index < 5; ++index) {
		gallery_points[index].x = landMark[index * 2];
		gallery_points[index].y = landMark[index * 2 + 1];
	}
	face_recognizer->ExtractFeatureWithCrop(img_data_color, gallery_points, feature_vec);
	return feature_vec;
}

double face_recognition_(const char* img1, const char* img2)
{
	if (FD == nullptr || face_recognizer == nullptr) {
		fprintf(stderr, "no find model!\n");
		return -2;
	}
	Mat image_1 = imread(img1);
	if (image_1.empty()) {
		fprintf(stderr, "load image fail!\n");
		return -3;
	}
	try {
		Mat dst;
		cvtColor(image_1, dst, COLOR_BGR2RGB);
		FD->run(dst.data, dst.cols, dst.rows);
	}
	catch (...) {
		fprintf(stderr, "path:%s\n", img1);
		if (FD->face_num() == 0)return -1;
		return -4;
	}
	image_enhancement(image_1, image_1, ImageProMethodData(EQUALIZATION_HSV));
	ImageData gallery_img_data_color(image_1.cols, image_1.rows, image_1.channels());
	gallery_img_data_color.data = image_1.data;
	vector<double> point = FD->finish();
	int pp1[4] = { int(point[5 * 2]),int(point[5 * 2 + 1]),int(point[5 * 2 + 2]),int(point[5 * 2 + 3]) };
	seeta::FacialLandmark gallery_points[5];
	for (int index = 0; index < 5; ++index) {
		gallery_points[index].x = point[index * 2];
		gallery_points[index].y = point[index * 2 + 1];
	}
	Mat image_2 = imread(img2);
	if (image_2.empty()) {
		fprintf(stderr, "load image fail!\n");
		return -5;
	}
	Mat dst;
	cvtColor(image_2, dst, COLOR_BGR2RGB);
	if (!FD->run(dst.data, dst.cols, dst.rows))
	{
		fprintf(stderr, "path:%s\n", img2);
		if (FD->face_num() == 0)return -1;
		return -6;
	}
	image_enhancement(image_2, image_2, ImageProMethodData(EQUALIZATION_LAB));
	ImageData probe_img_data_color(image_2.cols, image_2.rows, image_2.channels());
	probe_img_data_color.data = image_2.data;
	point = FD->finish();
	int pp2[4] = { int(point[5 * 2]),int(point[5 * 2 + 1]),int(point[5 * 2 + 2]),int(point[5 * 2 + 3]) };
	seeta::FacialLandmark probe_points[5];
	for (int index = 0; index < 5; ++index) {
		probe_points[index].x = point[index*2];
		probe_points[index].y = point[index*2 + 1];
	}
	rectangle(image_1, Rect(Point2i(pp1[0], pp1[1]),
		Point2i(pp1[2], pp1[3])), Scalar(0, 0, 255), 3);
	rectangle(image_2, Rect(Point2i(pp2[0], pp2[1]),
		Point2i(pp2[2], pp2[3])), Scalar(0, 0, 255), 3);
	for (int i = 0; i < 5; i++) {
		circle(image_1, Point(int(gallery_points[i].x), int(gallery_points[i].y)), 3, Scalar(0, 255, 0));
		circle(image_2, Point(int(probe_points[i].x), int(probe_points[i].y)), 3, Scalar(0, 255, 0));
	}
	imwrite("test_1.jpg", image_1);
	imwrite("test_2.jpg", image_2);
	float gallery_fea[2048];
	float probe_fea[2048];
	face_recognizer->ExtractFeatureWithCrop(gallery_img_data_color, gallery_points, gallery_fea);
	face_recognizer->ExtractFeatureWithCrop(probe_img_data_color, probe_points, probe_fea);

	// Caculate similarity of two faces
	float sim = face_recognizer->CalcSimilarity(gallery_fea, probe_fea);

	return double(sim);
}
Mat detect(vector<double>& point, const char * imgpath, double minarea, bool re_size)
{
	if (FD == nullptr) {
		fprintf(stderr, "no find model!\n");
		return Mat();
	}
	Mat image = imread(imgpath);
	if (image.empty()) {
		return Mat();
	}
	if (re_size) {
		resize(image, image, Size(image.cols / 2, image.rows / 2));
		image_enhancement(image, image, ImageProMethodData(EQUALIZATION_HSV));
	}
	Mat dst;
	cvtColor(image, dst, COLOR_BGR2RGB);
	if (FD->run(dst.data, dst.cols, dst.rows, minarea))
		FD->finish().swap(point);
	else
		point.clear();
	return image;
}
void detect(const Mat &src, vector<double> &point, double minarea, bool re_size)
{
	if (FD == nullptr) {
		fprintf(stderr, "no find model!\n");
		return;
	}
	Mat image = src.clone();
	if (image.empty()) {
		return;
	}
	if (re_size) {
		resize(image, image, Size(image.cols / 2, image.rows / 2));
		image_enhancement(image, image, ImageProMethodData(EQUALIZATION_HSV));
	}
	//image_enhancement(image, image, ImageProMethodData(EQUALIZATION_LAB));
	Mat RGBImage;
	cvtColor(image, RGBImage, COLOR_BGR2RGB);
	if (FD->run(RGBImage.data, RGBImage.cols, RGBImage.rows, minarea))
		FD->finish().swap(point);
	else
		vector<double>().swap(point);
}
void detect_s(const Mat &src, vector<vector<double>> &point, double minarea, bool re_size)
{
	if (FD == nullptr) {
		fprintf(stderr, "no find model!\n");
		return;
	}
	Mat image = src.clone();
	if (image.empty()) {
		return;
	}
	if (re_size) {
		resize(image, image, Size(image.cols / 2, image.rows / 2));
		image_enhancement(image, image, ImageProMethodData(EQUALIZATION_HSV));
	}
	Mat RGBImage;
	cvtColor(image, RGBImage, COLOR_BGR2RGB);
	if (FD->run(RGBImage.data, RGBImage.cols, RGBImage.rows, minarea, false)) {
		FD->finish_().swap(point);
		cout << point.size() << endl;
	}
	else
		point.clear();
}
void CreatMat(Mat & dst, unsigned char * src, int cols, int rows, int srider, int channel)
{
	int type = CV_8U;
	switch (channel)
	{
	case 1:type = CV_8UC1; break;
	case 2:type = CV_8UC2; break;
	case 3:type = CV_8UC3; break;
	case 4:type = CV_8UC4; break;
	default:
		break;
	}
	Mat(rows, cols, type, src, srider).copyTo(dst);
}
void facialPoseCorrection(const Mat & src, Mat & dst, int left_eye_x, int left_eye_y, int right_eye_x, int right_eye_y)
{
	float diffEyeX = right_eye_x - left_eye_x;
	float diffEyeY = right_eye_y - left_eye_y;

	float degree = 0.f;
	float pi = 3.1415926535897932384626433832795f;
	if (fabs(diffEyeX) >= 0.0000001f)
		degree = atanf(diffEyeY / diffEyeX) * 180.0f / pi;
	Mat rotate = getRotationMatrix2D(Point2f(diffEyeX, diffEyeY), degree, 1.0);
	warpAffine(src, dst, rotate, src.size(), 1, 0, Scalar(255, 255, 255));
}
double recoginizer(Mat & img1, Mat & img2, vector<double>& point1, vector<double>& point2)
{
	if (face_recognizer == nullptr)return -1;
	if (img1.empty() || point1.empty())return -2;
	if (img2.empty() || point2.empty())return -3;
	ImageData gallery_img_data_color(img1.cols, img1.rows, img1.channels());
	gallery_img_data_color.data = img1.data;
	seeta::FacialLandmark gallery_points[5];
	for (int index = 0; index < 5; ++index) {
		gallery_points[index].x = point1[index * 2];
		gallery_points[index].y = point1[index * 2 + 1];
	}
	ImageData probe_img_data_color(img2.cols, img2.rows, img2.channels());
	probe_img_data_color.data = img2.data;
	seeta::FacialLandmark probe_points[5];
	for (int index = 0; index < 5; ++index) {
		probe_points[index].x = point2[index * 2];
		probe_points[index].y = point2[index * 2 + 1];
	}
	float gallery_fea[2048];
	float probe_fea[2048];
	face_recognizer->ExtractFeatureWithCrop(gallery_img_data_color, gallery_points, gallery_fea);
	face_recognizer->ExtractFeatureWithCrop(probe_img_data_color, probe_points, probe_fea);

	// Caculate similarity of two faces
	float sim = face_recognizer->CalcSimilarity(gallery_fea, probe_fea);

	return double(sim);
}
double _stdcall distance(float fea_1[2048], float fea_2[2048])
{
	if (face_recognizer == nullptr)return -1;
	if (fea_1 == nullptr || fea_2 == nullptr)return -2;
	return double(face_recognizer->CalcSimilarity(fea_1, fea_2));
}
void recognition(Mat & img, vector<double>& point, float feature[2048])
{
	if (face_recognizer == nullptr)return;
	if (img.empty() || point.empty())return;
	ImageData gallery_img_data_color(img.cols, img.rows, img.channels());
	gallery_img_data_color.data = img.data;
	seeta::FacialLandmark gallery_points[5];
	for (int index = 0; index < 5; ++index) {
		gallery_points[index].x = point[index * 2];
		gallery_points[index].y = point[index * 2 + 1];
	}
	face_recognizer->ExtractFeatureWithCrop(gallery_img_data_color, gallery_points, feature);
}
unsigned char * _stdcall Imread(const char* filename, int & width, int & height, int & step)
{
	if (filename == nullptr) {
		if (imagedata != nullptr) {
			delete[]imagedata;
			imagedata = nullptr;
		}
		return nullptr;
	}
	IplImage* uu = cvLoadImage(filename);
	cv::Mat ss = cv::cvarrToMat(uu);
	if (imagedata != nullptr) {
		delete[]imagedata;
		imagedata = nullptr;
	}
	if (imagedata == nullptr) {
		imagedata = new unsigned char[ss.rows*ss.cols*ss.channels()];
	}
	memcpy(imagedata, ss.data, sizeof(unsigned char)*ss.rows*ss.cols*ss.channels());
	width = ss.cols;
	height = ss.rows;
	step = (int)ss.step;
	cvReleaseImage(&uu);
	//ss.release();
	return imagedata;
}
void Feature(const Mat &gallery_img_color, Rect rect, float fea[2048])
{
	cv::Mat gallery_img_gray;
	cv::cvtColor(gallery_img_color, gallery_img_gray, CV_BGR2GRAY);
	ImageData gallery_img_data_color(gallery_img_color.cols, gallery_img_color.rows, gallery_img_color.channels());
	gallery_img_data_color.data = gallery_img_color.data;
	FaceInfo gallery_faces;
	gallery_faces.bbox.x = rect.x;
	gallery_faces.bbox.y = rect.y;
	gallery_faces.bbox.width = rect.width;
	gallery_faces.bbox.height = rect.height;
	ImageData gallery_img_data_gray(gallery_img_gray.cols, gallery_img_gray.rows, gallery_img_gray.channels());
	gallery_img_data_gray.data = gallery_img_gray.data;
	seeta::FacialLandmark gallery_points[5];
	face_alignment->PointDetectLandmarks(gallery_img_data_gray, gallery_faces, gallery_points);
	face_recognizer->ExtractFeatureWithCrop(gallery_img_data_color, gallery_points, fea);
}