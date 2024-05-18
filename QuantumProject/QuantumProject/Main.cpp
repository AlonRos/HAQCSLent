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

#include <fstream>

using namespace std;

#define DEBUG3

#ifdef DEBUG
#define INPUT_TO_ALG "../../GUI/input_to_alg.txt"
#define OUTPUT_FROM_ALG "../../GUI/output_from_alg.txt"

#else
#define INPUT_TO_ALG "./input_to_alg.txt"
#define OUTPUT_FROM_ALG "./output_from_alg.txt"

#endif

int getAmountTime(int m1, int n1, int m2, int n2, int c, bool gpu) {
	int duration = 0;

	for (int i = 0; i < c; ++i) {
		Matrix2& mat1 = Matrix2::randomMatrix(m1, n1, 25);

		Matrix2& mat2 = Matrix2::randomMatrix(m2, n2, 25);

		auto start = chrono::high_resolution_clock::now();

		Matrix2& mat3 = Matrix2::mult(mat1, mat2, gpu);

		auto stop = chrono::high_resolution_clock::now();

		duration += duration_cast<chrono::milliseconds>(stop - start).count();

		delete& mat1;
		delete& mat2;
		delete& mat3;
	}

	return duration;
}

int main() {
#ifdef USEGPU
	init();
#endif

	int c = 2;
	int durationCPU, durationGPU;


	durationCPU = getAmountTime(1024, 1024, 1024, 8, c, false);
	durationGPU = getAmountTime(1024, 1024, 1024, 8, c, true);
	cout << "In average, the CPU calculated (1024 x 1024) * (1024 x 8) in " << (double)durationCPU / c << " milliseconds\n";
	cout << "In average, the GPU calculated (1024 x 1024) * (1024 x 8) in " << (double)durationGPU / c << " milliseconds\n\n";


	durationGPU = getAmountTime(1024, 1024, 1024, 4, c, true);
	durationCPU = getAmountTime(1024, 1024, 1024, 4, c, false);
	cout << "In average, the CPU calculated (1024 x 1024) * (1024 x 4) in " << (double)durationCPU / c << " milliseconds\n";
	cout << "In average, the GPU calculated (1024 x 1024) * (1024 x 4) in " << (double)durationGPU / c << " milliseconds\n\n";

	durationGPU = getAmountTime(1024, 1024, 1024, 1024, c, true);
	durationCPU = getAmountTime(1024, 1024, 1024, 1024, c, false);
	cout << "In average, the CPU calculated (1024 x 1024) * (1024 x 1024) in " << (double)durationCPU / c << " milliseconds\n";
	cout << "In average, the GPU calculated (1024 x 1024) * (1024 x 1024) in " << (double)durationGPU / c << " milliseconds\n\n";





}