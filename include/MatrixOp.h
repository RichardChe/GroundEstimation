#pragma once

#include <cstdint>

namespace MatrixOP
{
	float **allocateMatrix(int32_t nrow, int32_t ncol);
	void freeMatrix(float **m);
	void zeroMatrix(float** m, int32_t nrow, int32_t ncol);
	void printMatrix(float** m, int32_t nrow, int32_t ncol);
};