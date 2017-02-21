#define DllDemoAPI _declspec(dllexport)


#include "SYSUStereo.h"

using namespace cv;

Ptr<StereoSGBM> sgbm;

DllDemoAPI SYSUStereo::SYSUStereo(StereoParam _param)
{
	param.max_disp = _param.max_disp;
	param.p1 = _param.p1;
	param.p2 = _param.p2;
	param.win_size = _param.win_size;
	param.pre_filter_cap = _param.pre_filter_cap;

	int uniquenessRatio = 5;

	sgbm = StereoSGBM::create(0, param.max_disp, param.win_size, param.p1, param.p2, 0, 
		                         param.pre_filter_cap, uniquenessRatio, 0, 0, StereoSGBM::MODE_HH);
}

DllDemoAPI bool SYSUStereo::process(Mat left, Mat right)
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