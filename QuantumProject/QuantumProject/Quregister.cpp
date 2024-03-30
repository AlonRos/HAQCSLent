#include "Quregister.h"

Quregister::Quregister(int length, int num) : length(length) {
	int coordsLength = 2 << length;
	coords = new Matrix(coordsLength, 1);
}

void Quregister::applyGate(Matrix& gate) {
	Matrix::multIn(gate, *coords, *coords);
}