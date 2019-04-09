#ifndef __FACE_RECOGNOTION_H__
#define __FACE_RECOGNOTION_H__

#ifdef FACE_EXPORTS
#define FACE_API __declspec(dllexport)
#else
#define FACE_API __declspec(dllimport)
#endif

extern "C" FACE_API bool _stdcall create_load_models(const char* model_path);
extern "C" FACE_API void _stdcall del_model();
extern "C" FACE_API double _stdcall FaceRecognition(const char* img1, const char* img2);
extern "C" FACE_API double _stdcall Recoginizer(
	unsigned char* CameraForm,
	int FormWidth, int FormHeight, int FormStep,
	unsigned char* IDFace,
	int FaceWidth, int FaceHeight, int FaceStep
);
extern "C" FACE_API bool _stdcall drawFace(unsigned char* image, int width, int height, int step, int channels, int r, int g, int b, int minarea);
extern "C" FACE_API int _stdcall drawFaces(unsigned char* image, int width, int height, int step, int channels, int minarea, bool drawmark, bool rand_color, int r, int g, int b);
extern "C" FACE_API int* _stdcall findFace(unsigned char* image, int width, int height, int step, int channels,
	int &xLeft, int &yLeft, int &rectWitdh, int &rectHeight, int minarea); 
extern "C" FACE_API int* _stdcall findFace_(unsigned char* image, int width, int height, int step, int channels,
	int &xLeft, int &yLeft, int &xRight, int &yRight, int minarea);
extern "C" FACE_API bool _stdcall findFaceP(unsigned char* image, int width, int height, int step, int channels,
	int &xLeft, int &yLeft, int &rectWitdh, int &rectHeight, int *landMark, int minarea); 
extern "C" FACE_API bool _stdcall findFaceP_(unsigned char* image, int width, int height, int step, int channels,
	int &xLeft, int &yLeft, int &xRight, int &yRight, int *landMark, int minarea);
extern "C" FACE_API bool _stdcall findFaceRect(unsigned char* image, int width, int height, int step, int channels,
	int &xLeft, int &yLeft, int &rectWitdh, int &rectHeight, int minarea);
extern "C" FACE_API bool _stdcall findFacePoint(unsigned char* image, int width, int height, int step, int channels,
	int &xLeft, int &yLeft, int &xRight, int &yRight, int minarea);
extern "C" FACE_API int _stdcall landMarkLen();
extern "C" FACE_API int _stdcall featureVecLen();
extern "C" FACE_API unsigned int _stdcall featureVecLength();
extern "C" FACE_API bool _stdcall featureVec(
	unsigned char* image, int width, int height, int step, int channels, float *fea);
extern "C" FACE_API bool _stdcall featureVecPoint(
	unsigned char* image, int width, int height, int step, int channels, float *fea, int xLeft, int yLeft, int xRight, int yRight);
extern "C" FACE_API bool _stdcall featureVecRect(
	unsigned char* image, int width, int height, int step, int channels, float *fea, int xLeft, int yLeft, int rectWitdh, int rectHeight);
extern "C" FACE_API bool _stdcall featureVecP(
	unsigned char* image, int width, int height, int step, int channels, int *landMark, float *fea);
extern "C" FACE_API float* _stdcall featureVecs(
	unsigned char* image, int &len,
	int width, int height, int step, int channels); 
extern "C" FACE_API float* _stdcall _featureVec(
	unsigned char* image, int width, int height, int step, int channels, int *landMark);
extern "C" FACE_API double _stdcall distance(float fea_1[2048], float fea_2[2048]);
extern "C" FACE_API unsigned char * _stdcall Imread(const char* filename, int & width, int & height, int & step);

#endif // __FACE_RECOGNOTION_H__
