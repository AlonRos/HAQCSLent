#include "Quregister.h"
#include "Measurement.h"

Quregister::Quregister(int length, int num) : length(length) {
	coordsLength = 1 << length;
	coords = new Matrix2(coordsLength, 1);
	coords->entry(num, 0) = 1;
}

void Quregister::applyGate(Matrix2& gate) {
	Matrix2::multIn(gate, *coords, *coords);
}

int Quregister::getRegLength() {
	return length;
}

int Quregister::getCoordsLength() {
	return coordsLength;
}

Matrix2*& Quregister::getCoords() {
	return coords;
}

int Quregister::regMeasure(vector<Quregister> basis) {
	return measure(*this, basis);
}

int Quregister::regMeasureComputational() {
	return measureComputational(*this);
}