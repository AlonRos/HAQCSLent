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

	int c = 1;

	int size = 1024;
	
	int duration = 0;
	for (int i = 0; i < c; ++i) {
		Matrix2& m1 = Matrix2::randomMatrix(size, size, 25);

#ifdef DEBUG
		m1.print();
		cout << "\n";
#endif

		Matrix2& m2 = Matrix2::randomMatrix(size, size, 25);

#ifdef DEBUG
		m2.print();
		cout << "\n";
#endif


		auto start = chrono::high_resolution_clock::now();

		Matrix2& m = m1 * m2;

		auto stop = chrono::high_resolution_clock::now();
		duration += duration_cast<chrono::milliseconds>(stop - start).count();

#ifdef DEBUG
		m.print();
		cout << "\n";
		cout << "\n";
#endif
		delete& m1;
		delete& m2;
		delete& m;

	}

	cout << duration / c;
}