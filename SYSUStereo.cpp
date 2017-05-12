#include "SYSUStereo.h"

using namespace cv;

Ptr<StereoSGBM> sgbm;

SYSUStereo::SYSUStereo(StereoParam _param)
	:param(_param.mMaxDisp, _param.mP1, _param.mP2, _param.mWinSize, _param.mPreFilterCap)
{
	int uniquenessRatio = 5;

	sgbm = StereoSGBM::create(0, param.mMaxDisp, param.mWinSize, param.mP1, param.mP2, 0,
		                         param.mPreFilterCap, uniquenessRatio, 0, 0, StereoSGBM::MODE_HH);
}

bool SYSUStereo::process(Mat left, Mat right)
{
	resize(left, left, Size(left.cols / 2, left.rows / 2));
	resize(right, right, Size(right.cols / 2, right.rows / 2));

	Mat disp16s;
	sgbm->compute(left, right, disp16s);

	disp16s = disp16s / 16;
	disp16s.convertTo(disp, CV_8UC1);

	resize(disp, disp, Size(disp.cols * 2, disp.rows * 2));
	disp = disp * 2;

	return true;
}
