#define MTCNN_EXPORTS
#include "face_detection.h"

#define USE_SHELL_OPEN
#ifndef  nullptr
#define nullptr 0
#endif
#if defined(_MSC_VER)
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h> 
#else
#include <unistd.h>
#endif

#include <stdio.h>
using mtcnn::FaceDet;
bool FaceDet::creat_mtcnn(const char * model_path)
{
	if (mtcnn == nullptr)
		mtcnn = new MTCNN(model_path);
	if (!mtcnn->IsEnable()) {
		std::cerr << model_path << " not exist!" << std::endl;
		throw "no find model!";
	}
	return true;
}
void FaceDet::del_mtcnn()
{
	if (mtcnn != nullptr) {
		delete mtcnn;
		mtcnn = nullptr;
	}
}

vector<double> FaceDet::finish()
{
	if (finalBbox->empty()) return vector<double>();
	if (finalBbox->size() == 1) {
		vector<double> data(5 * 2 + 4);
		for (int i = 0; i < 5; i++) {
			data[i * 2] = double((*finalBbox)[0].ppoint[i]);
			data[i * 2 + 1] = double((*finalBbox)[0].ppoint[5 + i]);
		}
		data[5 * 2] = (*finalBbox)[0].x1;
		data[5 * 2 + 1] = (*finalBbox)[0].y1;
		data[5 * 2 + 2] = (*finalBbox)[0].x2;
		data[5 * 2 + 3] = (*finalBbox)[0].y2;
		return data;
	}
}

vector<vector<double>> mtcnn::FaceDet::finish_()
{	
	if (finalBbox->empty()) return vector<vector<double>>();
	//for (size_t i = 0; i < finalBbox->size() - 1; i++) {
	//	size_t max_index = i;
	//	for (size_t j = i + 1; j < finalBbox->size(); j++) {
	//		if ((*finalBbox)[max_index].score < (*finalBbox)[j].score) {
	//			max_index = j;
	//		}
	//	}
	//	Bbox temp = (*finalBbox)[i];
	//	(*finalBbox)[i] = (*finalBbox)[max_index];
	//	(*finalBbox)[max_index] = temp;
	//}
	vector<vector<double>> data;
	for (size_t i = 0; i < finalBbox->size(); i++) {
		vector<double> point(5 * 2 + 4);
		for (int j = 0; j < 5; j++) {
			point[j * 2] = double((*finalBbox)[i].ppoint[j]);
			point[j * 2 + 1] = double((*finalBbox)[i].ppoint[5 + j]);
		}
		point[5 * 2] = (*finalBbox)[i].x1;
		point[5 * 2 + 1] = (*finalBbox)[i].y1;
		point[5 * 2 + 2] = (*finalBbox)[i].x2;
		point[5 * 2 + 3] = (*finalBbox)[i].y2;
		data.push_back(point);
	}
	return data;
}


bool FaceDet::run(unsigned char * src, int cols, int rows, double area ,bool isOne)
{
	std::vector<Bbox>().swap(*finalBbox);
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(src, ncnn::Mat::PIXEL_RGB, cols, rows);
	if (isOne) { 
		mtcnn->detect(ncnn_img, *finalBbox);
		if(finalBbox->empty()) {
			return false;
			//fprintf(stderr, "error 1: Face Detection is empty!\n");
			//throw "error 1: Face Detection is empty!\n";
		}
		Bbox maxBbox;
		maxBbox.score = 0;
		maxBbox.area = 0;
		for (std::vector<Bbox>::iterator iter = finalBbox->begin(); iter != finalBbox->end(); ++iter) {
			if (iter->area < area)continue;
			if (maxBbox.score < iter->score && iter->area > maxBbox.area)
				maxBbox = *iter;
		}
		if (maxBbox.score == 0) {
			return false;
			//fprintf(stderr, "error 2: Face Detection is empty!\n");
			//throw "error 2: Face Detection is empty!\n";
		}
		*finalBbox = vector<Bbox>(1);
		(*finalBbox)[0] = maxBbox;
	}
	else mtcnn->detect(ncnn_img, *finalBbox);
	return true;
}