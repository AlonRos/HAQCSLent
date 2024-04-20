#include <iostream>
#include "Matrix.h"
#include "Log.h"
#include "Gates.h"
#include <random>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
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
	 int x = 3;


	 init();

	 Quregister r1(1, 0);

	 r1.applyGate(hadamard);

	 r1.getCoords()->print();

	 cout << r1.regMeasureComputational();
	
	
 }