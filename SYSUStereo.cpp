#define DllDemoAPI _declspec(dllexport)
#include "SYSUStereo.h"
#include <cstdio>

DllDemoAPI SYSUStereo::SYSUStereo(StereoParam _param)
{
	param.max_disp = _param.max_disp;
	param.p1 = _param.p1;
	param.p2 = _param.p2;
	param.win_size = _param.win_size;
}

DllDemoAPI bool SYSUStereo::process( Mat left,  Mat right)
{

	resize(left,left,Size(left.cols/2,left.rows/2));
	resize(right,right,Size(right.cols/2,right.rows/2));

	Mat disp_16s;
	int mindisparity = 0;
	int numDisparities = param.max_disp;
	int SADWindowSize = param.win_size;
	int P1 = param.p1;
	int P2 = param.p2;
	int preFilterCap = param.pre_filter_cap;

	StereoSGBM stereosgbm(mindisparity,numDisparities,SADWindowSize,P1,P2,0,preFilterCap);
	stereosgbm.fullDP = true;

	stereosgbm(left,right,disp_16s);

	disp_16s = disp_16s/16;
	disp_16s.convertTo(disp,CV_8UC1);

	resize(disp,disp,Size(right.cols*2,right.rows*2));
	disp = disp*2;

	return true;
}