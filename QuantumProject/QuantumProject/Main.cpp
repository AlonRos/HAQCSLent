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
#include "Grover'sAlgorithm.h"

using namespace std;

int main() {
	int N = 3000;

	int* f = new int[N];
	f[60] = 1;

	cout << grover(f, N);
}