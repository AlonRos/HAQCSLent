#include "Gates.h"

Matrix* bitFlipPtr;

Matrix* hadamardPtr;

Matrix* CNOTPtr;


void initializeGates() {
	complex_t bitFlipArr[2][2] = {
		{ 0, 1 },
		{ 1, 0 }
	};
	bitFlipPtr = &Matrix::fromArray(2, 2, (complex_t*) bitFlipArr);

	complex_t hadamardArr[2][2] = {
		{ 1 / sqrt(2), 1 / sqrt(2) },
		{ 1 / sqrt(2), -1 / sqrt(2) }
	};
	hadamardPtr = &Matrix::fromArray(2, 2, (complex_t*)hadamardArr);

	complex_t CNOTArr[4][4] = {
		{ 1, 0, 0, 0 },
		{ 0, 1, 0, 0 },
		{ 0, 0, 0, 1 },
		{ 0, 0, 1, 0 }
	};
	CNOTPtr = &Matrix::fromArray(4, 4, (complex_t*)CNOTArr);



}