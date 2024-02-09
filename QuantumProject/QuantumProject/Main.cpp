#include <iostream>
#include "Matrix.h"
#include "Gates.h"

using namespace std;

int main() {
	initializeGates();

	Matrix& bitFlip = *bitFlipPtr;
	Matrix& hadamard = *hadamardPtr;
	Matrix& CNOT = *CNOTPtr;


	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			cout << hadamard.entry(i, j) << " ";
		}
		cout << '\n';
	}
	cout << '\n';

	Matrix zero = Matrix::fromArray(2, 1, new complex_t[]{ {1}, {0} });

	Matrix superposition = hadamard * zero;

	for (int i = 0; i < 2; ++i) {
		cout << superposition.entry(i, 0) << " ";
	}

}