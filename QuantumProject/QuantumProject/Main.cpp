#include <iostream>
#include "Matrix.h"
#include "Log.h"
#include "Gates.h"
#include <random>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define DEBUG3
 
 using namespace std;
 
 int main() {
	srand(time(NULL));
	cudaEvent_t beg, end;
	cudaEventCreate(&beg);
	cudaEventCreate(&end);
	float elapsed_time;

#ifdef USEGPU // the first function of the gpu takes more time
	Matrix& mat1 = Matrix::randomMatrix(1, 1, 25);
	Matrix& mat2 = Matrix::randomMatrix(1, 1, 25);
	mat1 + mat2;
#endif
	
	int c = 1;

	int size = 1024;

	int duration = 0;
	for (int i = 0; i < c; ++i) {
		Matrix& m1 = Matrix::randomMatrix(size, size, 25);

#ifdef DEBUG
		m1.print();
		cout << "\n";
#endif
		
		Matrix& m2 = Matrix::randomMatrix(size, size, 25);

#ifdef DEBUG
		m2.print();
		cout << "\n";
#endif

		
		auto start = chrono::high_resolution_clock::now();
		//cudaEventRecord(beg);


		Matrix&m = m1 * m2;

		//cudaEventRecord(end);
		//cudaEventSynchronize(beg);
		//cudaEventSynchronize(end);
		//cudaEventElapsedTime(&elapsed_time, beg, end);

		//printf("aaaaaaaaaaaaaaa %f\n", elapsed_time);


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