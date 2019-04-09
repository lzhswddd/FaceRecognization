#define MAT_EXPORTS
#include "MatTools.h"
#include <windows.h>

using namespace cv;

//int main()
//{
//
//	Mat src;
//	Mat dst;
//	//屏幕截图
//	HBITMAP hBmp;
//	Screen(hBmp);
//
//	//类型转换
//	HBitmapToMat(hBmp, src);
//
//	//调整大小
//	resize(src, dst, Size(1200, 800), 0, 0);
//
//	imshow("dst", dst);
//	DeleteObject(hBmp);
//	waitKey();//这里调节帧数  现在200ms是5帧
//	return 0;
//}

//抓取当前屏幕函数
void Screen(HBITMAP& hBmp) {

	//创建画板
	HDC hScreen = CreateDC("DISPLAY", NULL, NULL, NULL);
	HDC	hCompDC = CreateCompatibleDC(hScreen);
	//取屏幕宽度和高度
	int		nWidth = GetSystemMetrics(SM_CXSCREEN);
	int		nHeight = GetSystemMetrics(SM_CYSCREEN);
	//创建Bitmap对象
	hBmp = CreateCompatibleBitmap(hScreen, nWidth, nHeight);
	HBITMAP hOld = (HBITMAP)SelectObject(hCompDC, hBmp);
	BitBlt(hCompDC, 0, 0, nWidth, nHeight, hScreen, 0, 0, SRCCOPY);
	SelectObject(hCompDC, hOld);
	//释放对象
	DeleteDC(hScreen);
	DeleteDC(hCompDC);
}


//把HBITMAP型转成Mat型
BOOL HBitmapToMat(HBITMAP& _hBmp, Mat& _mat)

{
	//BITMAP操作
	BITMAP bmp;
	GetObject(_hBmp, sizeof(BITMAP), &bmp);
	int nChannels = bmp.bmBitsPixel == 1 ? 1 : bmp.bmBitsPixel / 8;
	int depth = bmp.bmBitsPixel == 1 ? IPL_DEPTH_1U : IPL_DEPTH_8U;
	//mat操作
	Mat v_mat;
	v_mat.create(cvSize(bmp.bmWidth, bmp.bmHeight), CV_MAKETYPE(CV_8U, nChannels));
	GetBitmapBits(_hBmp, bmp.bmHeight*bmp.bmWidth*nChannels, v_mat.data);
	_mat = v_mat;
	return TRUE;
}

void getScreen(Mat & frame)
{
	//屏幕截图
	HBITMAP hBmp;
	Screen(hBmp);
	//类型转换
	HBitmapToMat(hBmp, frame);
	DeleteObject(hBmp);
}
