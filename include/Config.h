#pragma once
#include <string>
using namespace std;

namespace Config
{
	const string imageDir("/home/kitti_data/data/2010_03_09_drive_0019");
	const string imageBaseName[2] = { "I1_", "I2_" };
	const string imageFormat(".png");
	const int idxWidth = 6;
	const int frameBegin = 0;
	const int frameEnd = 300;

	const bool saveFile = false;
	const string imageSaveDir("Not exist");
}