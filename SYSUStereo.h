#include "cv.h"
#include "highgui.h"

using namespace cv;

#ifdef DllDemoAPI
#else
#define DllDemoAPI _declspec(dllimport)

#endif

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

