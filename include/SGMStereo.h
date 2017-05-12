#pragma once

#include <opencv2/opencv.hpp>

using namespace cv;

class SGMStereo
{
public:
	struct Parameter
	{
		int mMaxDisp;
		int mP1;
		int mP2;
		int mWinSize;
		int mPreFilterCap;
		bool mHalfResolution;
		Parameter(int maxDisp, int P1, int P2, int winSize, int preFilterCap,bool halfResolution)
		{
			mMaxDisp = maxDisp;
			mP1 = P1;
			mP2 = P2;
			mWinSize = winSize;
			mPreFilterCap = preFilterCap;
			mHalfResolution = halfResolution;
		}
		Parameter()
		{
			mMaxDisp = 64;
			mP1 = 100;
			mP2 = 2700;
			mWinSize = 4;
			mPreFilterCap = 180;
			mHalfResolution = true;
		}
	};
	bool process(const Mat& leftImg, const Mat& rightImg);
	SGMStereo(Parameter param);
	Mat getDisparity();
private:
	Mat mDisp;
	Parameter& mParam;
	Ptr<StereoSGBM> mSGBM;
};