#include "SGMStereo.h"

SGMStereo::SGMStereo(Parameter _param):
mParam(_param)
{
	int uniquenessRatio = 5;
	if (mParam.mHalfResolution)
		mSGBM = StereoSGBM::create(0, mParam.mMaxDisp / 2, mParam.mWinSize, mParam.mP1, mParam.mP2, 0,
			mParam.mPreFilterCap, uniquenessRatio, 0, 0, StereoSGBM::MODE_HH);
	else
		mSGBM = StereoSGBM::create(0, mParam.mMaxDisp, mParam.mWinSize, mParam.mP1, mParam.mP2, 0,
			mParam.mPreFilterCap, uniquenessRatio, 0, 0, StereoSGBM::MODE_HH);
}

bool SGMStereo::process(const Mat& left, const Mat& right)
{
	Mat disp16s;
	if (mParam.mHalfResolution)
	{
		Mat halfLeft, halfRight;
		resize(left, halfLeft, Size(left.cols / 2, left.rows / 2));
		resize(right, halfRight, Size(left.cols / 2, left.rows / 2));
		mSGBM->compute(halfLeft, halfRight, disp16s);
		disp16s *= 2;
		resize(disp16s, disp16s, Size(disp16s.cols * 2, disp16s.rows * 2));
	}
	else
		mSGBM->compute(left, right, disp16s);

	disp16s = disp16s / 16;
	disp16s.convertTo(mDisp, CV_8UC1);

	return true;
}

Mat SGMStereo::getDisparity() //TODO 这里最好加一个参数，表明输出什么格式的视差图（灰度，RGB）
{
	return mDisp;
}
