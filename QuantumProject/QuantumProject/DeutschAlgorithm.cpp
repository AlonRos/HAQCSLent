#include "DeutschAlgorithm.h"
#include "Gates.h"
#include <iostream>

int* generateBalancedFunction(int size) {
	 int* f = new int[size]();

	 int c = 0, index;
	 while (c < size / 2) {
		 do {
			 index = randBound(size);
		 } while (f[index] == 1);

		 f[index] = 1;
		 ++c;
	 }

	 return f;
}

int* generateConstantFunction(int size, int value) {
	if (value == 0) {
		return new int[size]();
	}
	else {
		int* f = new int[size];
		std::fill(f, f + size, 1);
		return f;
	}
}


Matrix2& createMatrixFromFunction(int* f, int matSize) {
	Matrix2& Uf = *new Matrix2(matSize, matSize);

	int amount = matSize / 2;
	for (int i = 0; i < amount; ++i) {
		Uf.entry(2 * i, (i << 1) | f[i]) = 1;
		Uf.entry(2 * i + 1, (i << 1) | (1 - f[i])) = 1;
	}

	return Uf;
}

bool isBalanced(int n, Matrix2& Uf) {
	int regLength = n + 1;

	Quregister q(regLength, 1);

	q.applyGateOnQubits(hadamard, 0, regLength);

	q.applyGate(Uf);

	q.applyGateOnQubits(hadamard, 1, regLength);

	bool balanced = q.regMeasureComputational(1, regLength) != 0;

	delete q.getCoords();

	return balanced;
}