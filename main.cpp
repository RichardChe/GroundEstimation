#include <opencv2/opencv.hpp>
// #include <cv.hpp>
// #include <highgui.hpp>
#include "SYSUStereo.h"
#include "SYSUGroundEstimator.h"
#include <iostream>
#include <time.h>

using namespace cv;
using namespace std;

//KITTI data main
//int main()
//{
//	int frameBegin = 180;
//	int frameEnd = 460;
//
//	//kitti original
//	char *leftDir  = "E:\\KITTIData\\00\\image_0\\%06d.png";
//	char *rightDir = "E:\\KITTIData\\00\\image_1\\%06d.png";
//	char *saveDir = "E:\\groundSeg-1-5\\%06d.png";
//	char *dispSaveDir = "E:\\groundSeg-1-5\\disp_%06d.png";
//	char *obsSaveDir = "E:\\groundSeg-1-5\\obs_%06d.png";
//
//	char leftFile[256];
//	char rightFile[256];
//	char groundFile[256];
//	char dispFile[256];
//	char obsFile[256];
//
//	Mat left_show;
//
//	StereoParam param;
//	param.max_disp = 64;
//
//	SYSUStereo sysuStereo(param);
//
//	Calib calib;
//
//	//kitti
//	double base = 0.53;
//	double f =  7.188560000000e+02;
//	double cu = 6.071928000000e+02;
//	double cv = 1.852157000000e+02;
//	double height = 1.65;
//
//	calib.f     = f;
//	calib.cu    = cu;
//	calib.cv    = cv;
//	calib.base  = base;
//	calib.vehicle_height = height;
//
//	ObsParam obsParam;
//	obsParam.maxZ = 100;
//	obsParam.minY = 0.3;
//
//	SYSUGroundEstimator groundEstimator(calib,obsParam);
//
//	Mat leftImg,rightImg,groundImg,disp;
//
//	for (int i = frameBegin; i <= frameEnd; i++)
//	{
//		cout<<"frame:"<<i<<endl;
//
//		sprintf(leftFile,leftDir,i);
//		sprintf(rightFile,rightDir,i);
//		//sprintf(groundFile,saveDir,i);
//		//sprintf(dispFile,dispSaveDir,i);
//
//		leftImg = imread(leftFile,CV_LOAD_IMAGE_GRAYSCALE);
//		rightImg = imread(rightFile,CV_LOAD_IMAGE_GRAYSCALE);
//
//		groundEstimator.compute(leftImg,rightImg,groundImg,disp);
//
//		imshow("ground",groundImg);
//		imshow("disp", disp);
//
//		waitKey();
//	}
//
//	return 0;
//}

//tongji data main
int main()
{
	int frameBegin = 80;  //180-460
	int frameEnd = 200;

	//kitti original
	//char *leftDir = "E:\\data-15-12-25\\rectified\\left_%06d.jpg";
	//char *rightDir = "E:\\data-15-12-25\\rectified\\right_%06d.jpg";
	//char *saveDir = "E:\\groundSeg-1-5\\%06d.png";
	//char *dispSaveDir = "E:\\groundSeg-1-5\\disp_%06d.png";
	//char *obsSaveDir = "E:\\groundSeg-1-5\\obs_%06d.png";

	//tongji 2016-04-15
	//char *leftDir = "E:\\data-15-12-25\\rectified\\left_%06d.jpg";
	//char *rightDir = "E:\\data-15-12-25\\rectified\\right_%06d.jpg";

	//tongji 2016-04-15
	char *leftDir = "/home/lenovo/lane test/left%06d.jpg";
	char *rightDir = "/home/lenovo/lane test/right%06d.jpg";

	char leftFile[256];
	char rightFile[256];
	char groundFile[256];
	char dispFile[256];
	char obsFile[256];

	Mat left_show;

	StereoParam param;
	param.mMaxDisp = 64;

	SYSUStereo sysuStereo(param);

	Calib calib;
	GridParam gridParam;
	gridParam.gridSize = 0.2;
	gridParam.maxX = 20;
	gridParam.maxZ = 50;

	// tongji
	calib.f = 1362.4194 / 2;
	calib.cu = 450;
	calib.cv = 180;
	//calib.cu    = 968.00564;
	//calib.cv    = 230;
	calib.base = 0.4;
	calib.vehicle_height = 1.5;

	ObsParam obsParam;
	obsParam.maxZ = 100;
	obsParam.minY = 0.3;

	SYSUGroundEstimator groundEstimator(calib,obsParam,gridParam);

	Mat leftImg,rightImg,groundImg,disp;

	for (int i = frameBegin; i <= frameEnd; i++)
	{

		cout<<"frame:"<<i<<endl;

		sprintf(leftFile,leftDir,i);
		sprintf(rightFile,rightDir,i);
		//sprintf(groundFile,saveDir,i);
		//sprintf(dispFile,dispSaveDir,i);

		leftImg = imread(leftFile,CV_LOAD_IMAGE_GRAYSCALE);
		rightImg = imread(rightFile,CV_LOAD_IMAGE_GRAYSCALE);

		leftImg = leftImg(Rect(0, 100, 800, 300));
		rightImg = rightImg(Rect(0, 100, 800, 300));

		imshow("left", leftImg);
		imshow("right", rightImg);
		//waitKey();
		//resize(leftImg, leftImg, Size(leftImg.cols / 2, leftImg.rows / 2));
		//resize(rightImg, rightImg, Size(rightImg.cols / 2, rightImg.rows / 2));
		time_t begintime, endtime;
		begintime = clock();
		groundEstimator.compute(leftImg,rightImg,groundImg,disp);
		endtime = clock();
		cout << "time:" << endtime - begintime << endl;
		imshow("ground",groundImg);
		imshow("disp", disp);

		Mat gridShow;

		imwrite("grid.png", groundEstimator.gridMap_);
		resize(groundEstimator.gridMap_, gridShow, Size(groundEstimator.gridMap_.cols*2, groundEstimator.gridMap_.rows*2));

		imshow("grid", gridShow);
		// if (i == frameBegin)
		// 	waitKey();
		//FileStorage fs(".\\grid.xml", FileStorage::WRITE);
		//fs << "grid" << groundEstimator.gridMap_;
		//fs.release();
		waitKey(10);
	}

	return 0;
}
