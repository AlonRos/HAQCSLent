#include "Gates.h"

static complex_t bitFlipArr[2][2] = {
	{ 0, 1 },
	{ 1, 0 }
};
Matrix& bitFlip = Matrix::fromArray(2, 2, (complex_t*)bitFlipArr);

static complex_t hadamardArr[2][2] = {
	{ 1 / sqrt(2), 1 / sqrt(2) },
	{ 1 / sqrt(2), -1 / sqrt(2) }
};
Matrix& hadamard = Matrix::fromArray(2, 2, (complex_t*)hadamardArr);

static complex_t CNOTArr[4][4] = {
	{ 1, 0, 0, 0 },
	{ 0, 1, 0, 0 },
	{ 0, 0, 0, 1 },
	{ 0, 0, 1, 0 }
};
Matrix& CNOT = Matrix::fromArray(4, 4, (complex_t*)CNOTArr);
