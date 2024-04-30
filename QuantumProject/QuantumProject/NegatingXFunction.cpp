#include "NegatingXFunction.h"
#include "Gates.h"

Matrix2& U(int num) {
	return *new Matrix2(4, 4, (complex_t*)&UxArr[num], true, 4);
}

int findNum(Matrix2& Ux) {
	Quregister q(2, 0);

	q.applyGateOnQubits(hadamard, 0, 2);

	q.applyGate(Ux);

	q.applyGateOnQubits(hadamard, 0, 2);

	q.applyGate(U(0) * -1);

	q.applyGateOnQubits(hadamard, 0, 2);

	int num = q.regMeasureComputational();

	delete q.getCoords();

	return num;
}