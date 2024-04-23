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


using namespace std;

//#define DEBUG

int main() {
#ifdef USEGPU
	init();
#endif

	int c = 10;

	int size = 1024;

	for (int N = 1; N < 1000; ++N) {

		int durationC = 0, durationG = 0;
		for (int i = 0; i < c; ++i) {
			Matrix2& m1 = Matrix2::randomMatrix(N, N, 25);

			Matrix2& m2 = Matrix2::randomMatrix(N, N, 25);

			auto start = chrono::high_resolution_clock::now();

			Matrix2& m = Matrix2::mult(m1, m2, false);

			auto stop = chrono::high_resolution_clock::now();
			durationC += duration_cast<chrono::microseconds>(stop - start).count();

			delete& m1;
			delete& m2;
			delete& m;
		}

		for (int i = 0; i < c; ++i) {
			Matrix2& m1 = Matrix2::randomMatrix(N, N, 25);

			Matrix2& m2 = Matrix2::randomMatrix(N, N, 25);

			auto start = chrono::high_resolution_clock::now();

			Matrix2& m = Matrix2::mult(m1, m2, true);

			auto stop = chrono::high_resolution_clock::now();
			durationG += duration_cast<chrono::microseconds>(stop - start).count();

			delete& m1;
			delete& m2;
			delete& m;
		}

		if (durationC < durationG) {
			cout << "c " << N << " " << durationC << " " << durationG << '\n';
		}
		else {
			cout << "g " << N << " " << durationC << " " << durationG << '\n';

		}
	}
}