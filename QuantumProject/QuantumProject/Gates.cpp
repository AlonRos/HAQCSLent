#include "Gates.h"

static complex_t bitFlipArr[2][2] = {
	{ 0, 1 },
	{ 1, 0 }
};
Matrix2& bitFlip = *new Matrix2(2, 2, (complex_t*)bitFlipArr, true, 2);

static complex_t hadamardArr[2][2] = {
	{ 1 / sqrt(2), 1 / sqrt(2) },
	{ 1 / sqrt(2), -1 / sqrt(2) }
};
Matrix2& hadamard = *new Matrix2(2, 2, (complex_t*)hadamardArr, true, 2);

static complex_t CNOTArr[4][4] = {
	{ 1, 0, 0, 0 },
	{ 0, 1, 0, 0 },
	{ 0, 0, 0, 1 },
	{ 0, 0, 1, 0 }
};
Matrix2& CNOT = *new Matrix2(4, 4, (complex_t*)CNOTArr, true, 4);

Matrix2& phaseShift(double phi) {
	complex_t phaseShiftArr[2][2] = {
		{ 1, 0},
		{ 0, complex_t(cos(phi), sin(phi))}
	};

	return *new Matrix2(2, 2, (complex_t*)phaseShiftArr);
}