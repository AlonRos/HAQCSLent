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
	int n = 10;

	int size = 1 << n;

	int* f = generateBalancedFunction(size);

	//int* f = generateConstantFunction(size, 1);

	Matrix2& Uf = createMatrixFromFunction(f, 2 * size);

	free(f);

	cout << boolalpha << isBalanced(n, Uf);

	delete& Uf;

}