#include <iostream>
#include "Log.h"
#include "Matrix2.h"
#include "Gates.h"
#include <random>
#include <chrono>
#include "Quregister.h"

#ifdef USEGPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaHeader.cuh"
#endif

#include "NegatingXFunction.h"
#include "DeutschAlgorithm.h"

using namespace std;

int main() {
	Quregister q(10, 0);

	q.applyGates({ { Matrix2::kronecker(hadamard, hadamard), 1, 3 }, { Matrix2::kronecker(hadamard, Matrix2::kronecker(hadamard, hadamard)), 1, 4 } });

	q.getCoords()->print();
}