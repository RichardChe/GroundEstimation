#include <opencv2/opencv.hpp>

using namespace cv;

struct StereoParam
{
	int mMaxDisp;
	int mP1;
	int mP2;
	int mWinSize;
	int mPreFilterCap;
	StereoParam(int maxDisp, int P1 ,int P2, int winSize, int preFilterCap)
	{
		mMaxDisp = maxDisp;
		mP1 = P1;
		mP2 = P2;
		mWinSize = winSize;
		mPreFilterCap = preFilterCap;
	}
};

class SYSUStereo
{
public:
	SYSUStereo(StereoParam _param);
	Mat disp;
	StereoParam param;
	bool process(Mat left, Mat right);
};
