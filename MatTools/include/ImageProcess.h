#ifndef __IMAGEPROCESS_H__
#define __IMAGEPROCESS_H__

#include <opencv2/opencv.hpp>
using cv::Mat;
using cv::Size;

//High-Frequency-Emphasis Filters
Mat Butterworth_Homomorphic_Filter(Size sz, float D, float n, float high_h_v_TB, float low_h_v_TB, Mat& realIm);
Mat Butterworth_Filter(Size sz, float D, float n, float high_h_v_TB, float low_h_v_TB, Mat& realIm);
Mat Filter(Size sz, float D, float n, float high_h_v_TB, float low_h_v_TB, Mat& realIm);
//DFT ·µ»Ø¹¦ÂÊÆ×Power
Mat Fourier_Transform(Mat frame_bw, Mat& image_complex, Mat &image_phase, Mat &image_mag);
void Inv_Fourier_Transform(Mat input, Mat& inverseTransform);
//SHIFT
void Shifting_DFT(Mat &fImage);
void homomorphicFiltering(Mat src, Mat& dst, float D0, float n, float rL, float rH);

#endif