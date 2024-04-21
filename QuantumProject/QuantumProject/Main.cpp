#include <iostream>
#include "Matrix.h"
#include "Log.h"
#include "Gates.h"
#include <random>
#include <chrono>

#ifdef USEGPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#endif

#include "Quregister.h"

 
 using namespace std;
 
 void init() {
	 srand(time(NULL));

#ifdef USEGPU // the first function in the gpu takes more time
	 Matrix& mat1 = Matrix::randomMatrix(1, 1, 25);
	 Matrix& mat2 = Matrix::randomMatrix(1, 1, 25);
	 mat1 * mat2;
#endif
 }

 int main() {
	 init();

	 Quregister r1(1, 0);

	 int c = 1e5;
	 int duration = 0;
	 int amount1 = 0;

	 for (int i = 0; i < c; ++i) {

		 auto start = chrono::high_resolution_clock::now();

		 r1.applyGate(hadamard);

		 auto stop = chrono::high_resolution_clock::now();

		 duration += duration_cast<chrono::milliseconds>(stop - start).count();

		 amount1 += r1.regMeasureComputational();

	 }

	 cout << (double)amount1 / c << " " << (double) duration / c;


	
 }