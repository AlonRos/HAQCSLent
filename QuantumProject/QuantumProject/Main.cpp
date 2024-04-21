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
	 mat1 + mat2;
#endif
 }

 int main() {
	 init();

	 Quregister r1(1, 0);

	 int c = 2;
	 int duration = 0;
	 int amount1 = 0;

	 for (int N = 750; N < 1024; ++N) {
		 int durationG = 0, durationC = 0;

		 for (int i = 0; i < c; ++i) {
			 Matrix& m1 = Matrix::randomMatrix(N, N, 25);

			 Matrix& m2 = Matrix::randomMatrix(N, 1, 25);

			 auto start = chrono::high_resolution_clock::now();
			 Matrix::mult(m1, m2, true);
			 auto stop = chrono::high_resolution_clock::now();
			 durationG += duration_cast<chrono::milliseconds>(stop - start).count();

			 start = chrono::high_resolution_clock::now();
			 Matrix::mult(m1, m2, false);
			 stop = chrono::high_resolution_clock::now();
			 durationC += duration_cast<chrono::milliseconds>(stop - start).count();

			 //amount1 += r1.regMeasureComputational();

		 }

		 if (durationC < durationG) {
			 cout << "y " << N << '\n';
		 }

		 if (durationC > durationG) {
			 cout << "n " << N << '\n';
		 }
	 }




	 cout << (double)amount1 / c << " " << (double) duration / c;


	
 }