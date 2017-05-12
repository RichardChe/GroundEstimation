#include "MatrixOp.h"

#include <cstdlib>
#include <iostream>

float ** MatrixOP::allocateMatrix(int32_t nrow, int32_t ncol) {
	float **m;
	m = (float**)malloc(nrow*sizeof(float*));
	m[0] = (float*)calloc(nrow*ncol, sizeof(float));
	for (int32_t i = 1; i < nrow; i++) m[i] = m[i - 1] + ncol;
	return m;
}

void MatrixOP::freeMatrix(float **m) {
	free(m[0]);
	free(m);
}

void MatrixOP::zeroMatrix(float** m, int32_t nrow, int32_t ncol) {
	for (int32_t i = 0; i < nrow; i++)
		for (int32_t j = 0; j < ncol; j++)
			m[i][j] = 0;
}

void MatrixOP::printMatrix(float** m, int32_t nrow, int32_t ncol) {
	for (int32_t i = 0; i < nrow; i++) {
		for (int32_t j = 0; j < ncol; j++)
			std::cout << m[i][j] << " ";
		std::cout << std::endl;
	}
}