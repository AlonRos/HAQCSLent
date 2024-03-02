#include "Quregister.h"

Quregister::Quregister(int length, int num) : length(length) {
	int coordsLength = 2 << length;
	coords = new Matrix(coordsLength, 1);
}

void Quregister::applyGate(int i, int j, Matrix& gate) {
	coords = &(gate * coords->rows(i, j));
}