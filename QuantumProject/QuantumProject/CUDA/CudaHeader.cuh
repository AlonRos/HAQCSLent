#pragma once

#include "../Include/Matrix2.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>

#define MAX_BLOCK_SIZE 16

__host__
inline cuDoubleComplex complexToCudaComplex(complex_t z) {
	return make_cuDoubleComplex(z.real(), z.imag());
}

__host__
inline complex_t cudaComplexToComplex(cuDoubleComplex z) {
	return complex_t(z.x, z.y);
}

__host__
void init();