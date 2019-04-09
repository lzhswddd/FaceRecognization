#ifndef __FACEDETECTION_H__
#define __FACEDETECTION_H__
#ifdef MTCNN_EXPORTS
    #define  MTCNN_API __declspec(dllexport)
#else
    #define  MTCNN_API __declspec(dllimport)
#endif

#include "mtcnn.h"

using std::vector;
namespace mtcnn{
class FaceDet
{
public:
	MTCNN_API FaceDet() :finalBbox(new std::vector<Bbox>()){
		mtcnn = nullptr;
	}
	MTCNN_API ~FaceDet() {
		if (finalBbox != nullptr) {
			std::vector<Bbox>().swap(*finalBbox);
			delete finalBbox;
		}
		del_mtcnn();
	}

	MTCNN_API bool creat_mtcnn(const char * model_path);
	MTCNN_API void del_mtcnn();
	MTCNN_API bool run(unsigned char * src, int cols, int rows, double area = 0, bool isOne = true);
	MTCNN_API vector<double> finish();
	MTCNN_API vector<vector<double>> finish_();
	MTCNN_API int face_num() { return int(finalBbox->size()); }
private:
	std::vector<Bbox> *finalBbox;
	MTCNN *mtcnn;
};
}
#endif //__FACEDETECTION_H__