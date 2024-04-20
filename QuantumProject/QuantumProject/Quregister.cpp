#include "Quregister.h"
#include "Measurement.h"

Quregister::Quregister(int length, int num) : length(length) {
	coordsLength = 1 << length;
	coords = new Matrix(coordsLength, 1);
	coords->entry(num, 0) = 1;
}

void Quregister::applyGate(Matrix& gate) {
	Matrix::multIn(gate, *coords, *coords);
}

int Quregister::getRegLength() {
	return length;
}

int Quregister::getCoordsLength() {
	return coordsLength;
}

Matrix* Quregister::getCoords() {
	return coords;
}

int Quregister::regMeasure(vector<Quregister> basis) {
	return measure(*this, basis);
}

int Quregister::regMeasureComputational() {
	return measureComputational(*this);
}