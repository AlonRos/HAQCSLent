#include <iostream>
#include "Log.h"
#include "Matrix2.h"
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

#ifdef USEGPU // the first function in the gpu takes more time
	 Matrix& mat1 = Matrix::randomMatrix(16, 16, 25);
	 Matrix& mat2 = Matrix::randomMatrix(16, 16, 25);
	 mat1 * mat2;
#endif
 }

//#define DEBUG

 int main() {
	 init();

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

		 free(&m);
	 }

	 cout << duration / c;
}