#include <opencv2/opencv.hpp>
#include "SYSUStereo.h"
using namespace cv;

int main()
{
  Mat imLeft,imRight;

  imLeft  = imread("/home/lenovo/2010_03_09_drive_0019/I1_000000.png");
  imRight = imread("/home/lenovo/2010_03_09_drive_0019/I2_000000.png");

  StereoParam stereoParam(64,100,2700,4,180);
  SYSUStereo sysuStereo(stereoParam);

  sysuStereo.process(imLeft,imRight);

  imshow("disp",sysuStereo.disp);
  waitKey();

  return 0;
}
