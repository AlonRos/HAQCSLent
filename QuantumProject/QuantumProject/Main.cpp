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


int main() {
#ifdef USEGPU
	init();
#endif

	int c = 10;

	int duration = 0;
	
	int N = 1024;

	for (int i = 0; i < c; ++i) {
		Matrix2& m1 = Matrix2::randomMatrix(N, N, 25);
		//m1.print();
		//cout << "\n";

		Matrix2& m2 = Matrix2::randomMatrix(N, 8, 25);
		//m2.print();
		//cout << "\n";


		auto start = chrono::high_resolution_clock::now();

		Matrix2& m = Matrix2::mult(m1, m2);

		auto stop = chrono::high_resolution_clock::now();
		//m.print();
		//cout << "\n";

		duration += duration_cast<chrono::milliseconds>(stop - start).count();

		delete& m1;
		delete& m2;
		delete& m;
	}

	cout << (double)duration / c;

}