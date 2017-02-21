#ifdef DllDemoAPI
#else
#define DllDemoAPI _declspec(dllimport)

//extern"C"{
//DllDemoAPI bool _stdcall disp2Color(uint8_t* disp ,uint8_t* disparityColor,int width,int heigh, int maxdisp);
//DllDemoAPI int _stdcall subtract(int a, int b);
//DllDemoAPI int _stdcall multiple(int a, int b);
//}
#endif

#include "cv.hpp"
#include "highgui.hpp"

using namespace cv;

struct StereoParam
{
	int max_disp;
	int p1;
	int p2;
	int win_size;
	int pre_filter_cap;
};

class DllDemoAPI SYSUStereo
{
public:
	SYSUStereo(StereoParam);
	Mat disp;
	StereoParam param;
	bool process(Mat left, Mat right);
};