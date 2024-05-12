#include "Grover'sAlgorithm.h"
#include "Quregister.h"
#include "Gates.h"
#define _USE_MATH_DEFINES
#include <math.h>

static void applyUf(Quregister& q, int* f, int N) {
	for (int i = 0; i < N; ++i) {
		if (f[i] == 1) {
			q.getCoords()->entry(i, 0) *= -1;
		}
	}
}

int grover(int* f, int N) {
	int times = M_PI_4 * sqrt(N);

	Quregister q((int)ceil(log2(N)), 0);

	q.applyGateOnQubits(hadamard, 0, q.getRegLength());

	for (int i = 0; i < times; ++i) {
		applyUf(q, f, N);

		q.applyGateOnQubits(hadamard, 0, q.getRegLength());

		q.getCoords()->entry(0, 0) *= -1;

		q.applyGateOnQubits(hadamard, 0, q.getRegLength());
	}

	int res = q.regMeasureComputational();
	delete q.getCoords();
	return res;
}