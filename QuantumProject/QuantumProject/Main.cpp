#include <iostream>
#include "Matrix.h"
#include "Gates.h"
#include <random>
#include <chrono>

using namespace std;

int main() {
	initializeGates();

	Matrix& bitFlip = *bitFlipPtr;
	Matrix& hadamard = *hadamardPtr;
	Matrix& CNOT = *CNOTPtr;

	srand(time(NULL));

	int c = 20;

	int duration = 0;
	for (int i = 0; i < c; ++i) {
		Matrix& m1 = Matrix::randomMatrix(100, 100, 25);
		//m1.print();
		//cout << "\n";

		Matrix& m2 = Matrix::randomMatrix(100, 100, 25);
		//m2.print();
		//cout << "\n";
// 
		//cout << "\n";

		auto start = chrono::high_resolution_clock::now();
		m1 * m2;
		auto stop = chrono::high_resolution_clock::now();
		duration += duration_cast<chrono::microseconds>(stop - start).count();
	}

	cout << duration / c;

}