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

void Quregister::applyGateOnQubit(Matrix2& gate, int index) {
	applyGateOnQubits(gate, index, index + 1);
}

void Quregister::applyGates(Matrix2* gates, int i, int j) {
	Matrix2& newCoords = *new Matrix2(coordsLength, 1);

	Matrix2* coordsArr[2] = { coords, &newCoords };


	int amountGates = j - i;
	Matrix2* currentCoords = &newCoords, *nextCoords = coords;


	for (int index = i; index < j; ++index) {
		currentCoords = coordsArr[(index - i) % 2];
		nextCoords = coordsArr[(index - i + 1) % 2];
		nextCoords->zero();

		Matrix2& gate = gates[index - i];

		Matrix2* cols[2] = { &gate.col(0), &gate.col(1) };
		Matrix2* col;

		for (int i = 0; i < coordsLength; ++i) {
			if (currentCoords->entry(i, 0) != (complex_t)0) {
				int qb = (i >> index) & 1;

				col = cols[qb];

				nextCoords->entry(i, 0) += currentCoords->entry(i, 0) * col->entry(qb, 0);
				nextCoords->entry(i ^ (1 << index), 0) += currentCoords->entry(i, 0) * col->entry(1 - qb, 0);
			}
		}

	}

	if (currentCoords == &newCoords) {
		free(currentCoords);
	}

	coords = nextCoords;
}

void Quregister::applyGateOnQubits(Matrix2& gate, int i, int j) {
	Matrix2& newCoords = *new Matrix2(coordsLength, 1);

	Matrix2* coordsArr[2] = { coords, &newCoords };

	Matrix2* currentCoords = &newCoords, * nextCoords = coords;


	for (int index = i; index < j; ++index) {
		currentCoords = coordsArr[(index - i) % 2];
		nextCoords = coordsArr[(index - i + 1) % 2];
		nextCoords->zero();

		Matrix2* cols[2] = { &gate.col(0), &gate.col(1) };
		Matrix2* col;

		for (int i = 0; i < coordsLength; ++i) {
			if (currentCoords->entry(i, 0) != (complex_t)0) {
				int qb = (i >> index) & 1;

				col = cols[qb];

				nextCoords->entry(i, 0) += currentCoords->entry(i, 0) * col->entry(qb, 0);
				nextCoords->entry(i ^ (1 << index), 0) += currentCoords->entry(i, 0) * col->entry(1 - qb, 0);
			}
		}

	}

	if (currentCoords == &newCoords) {
		free(currentCoords);
	}

	coords = nextCoords;
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