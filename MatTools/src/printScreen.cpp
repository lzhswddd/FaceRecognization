#define MAT_EXPORTS
#include "MatTools.h"
#include <windows.h>

using namespace cv;

//int main()
//{
//
//	Mat src;
//	Mat dst;
//	//��Ļ��ͼ
//	HBITMAP hBmp;
//	Screen(hBmp);
//
//	//����ת��
//	HBitmapToMat(hBmp, src);
//
//	//������С
//	resize(src, dst, Size(1200, 800), 0, 0);
//
//	imshow("dst", dst);
//	DeleteObject(hBmp);
//	waitKey();//�������֡��  ����200ms��5֡
//	return 0;
//}

//ץȡ��ǰ��Ļ����
void Screen(HBITMAP& hBmp) {

	//��������
	HDC hScreen = CreateDC("DISPLAY", NULL, NULL, NULL);
	HDC	hCompDC = CreateCompatibleDC(hScreen);
	//ȡ��Ļ��Ⱥ͸߶�
	int		nWidth = GetSystemMetrics(SM_CXSCREEN);
	int		nHeight = GetSystemMetrics(SM_CYSCREEN);
	//����Bitmap����
	hBmp = CreateCompatibleBitmap(hScreen, nWidth, nHeight);
	HBITMAP hOld = (HBITMAP)SelectObject(hCompDC, hBmp);
	BitBlt(hCompDC, 0, 0, nWidth, nHeight, hScreen, 0, 0, SRCCOPY);
	SelectObject(hCompDC, hOld);
	//�ͷŶ���
	DeleteDC(hScreen);
	DeleteDC(hCompDC);
}


//��HBITMAP��ת��Mat��
BOOL HBitmapToMat(HBITMAP& _hBmp, Mat& _mat)

{
	//BITMAP����
	BITMAP bmp;
	GetObject(_hBmp, sizeof(BITMAP), &bmp);
	int nChannels = bmp.bmBitsPixel == 1 ? 1 : bmp.bmBitsPixel / 8;
	int depth = bmp.bmBitsPixel == 1 ? IPL_DEPTH_1U : IPL_DEPTH_8U;
	//mat����
	Mat v_mat;
	v_mat.create(cvSize(bmp.bmWidth, bmp.bmHeight), CV_MAKETYPE(CV_8U, nChannels));
	GetBitmapBits(_hBmp, bmp.bmHeight*bmp.bmWidth*nChannels, v_mat.data);
	_mat = v_mat;
	return TRUE;
}

void getScreen(Mat & frame)
{
	//��Ļ��ͼ
	HBITMAP hBmp;
	Screen(hBmp);
	//����ת��
	HBitmapToMat(hBmp, frame);
	DeleteObject(hBmp);
}
