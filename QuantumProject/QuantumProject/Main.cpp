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

#include "NegatingXFunction.h"

using namespace std;

int main() {
	int n = 5;

	Quregister q(n + 1, 1);

	int coordsLength = q.getCoordsLength(), regLength = n + 1;

	q.applyGateOnQubits(hadamard, 0, regLength);

	Matrix2& Uf = *new Matrix2(coordsLength, coordsLength);

	int* f = new int[1 << n];

	for (int i = 0; i < coordsLength / 8; ++i) {
		f[i] = 1;
	}

	for (int i = coordsLength / 8; i < coordsLength / 4; ++i) {
		f[i] = 0;

	}

	for (int i = coordsLength / 4; i < 3 * coordsLength / 8; ++i) {
		f[i] = 0;

	}

	for (int i = 3 * coordsLength / 8; i < coordsLength / 2; ++i) {
		f[i] = 1;
	}

	for (int i = 0; i < coordsLength / 2; ++i) {
		Uf.entry(2 * i, (i << 1) | f[i]) = 1;
		Uf.entry(2 * i + 1, (i << 1) | (1 - f[i])) = 1;
	}

	q.applyGate(Uf);

	q.applyGateOnQubits(hadamard, 1, regLength);

	cout << q.regMeasureComputational(1, regLength);

}