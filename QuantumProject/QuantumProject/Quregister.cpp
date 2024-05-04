#include "Quregister.h"
#include "Measurement.h"

#define _USE_MATH_DEFINES
#include <math.h>

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

void Quregister::applyGates(Matrix2* gates, int beginIndex, int endIndex) {
	Matrix2* newCoords = new Matrix2(coordsLength, 1);

	Matrix2* coordsArr[2] = { coords, newCoords };

	Matrix2* currentCoords = newCoords, *nextCoords = coords;

	int amountGates = endIndex - beginIndex;

	for (int index = beginIndex; index < endIndex; ++index) {
		currentCoords = coordsArr[(index - beginIndex) % 2];
		nextCoords = coordsArr[(index - beginIndex + 1) % 2];
		nextCoords->zero();

		Matrix2& gate = gates[index - beginIndex];

		Matrix2* cols[2] = { &gate.col(0), &gate.col(1) };
		Matrix2* col;

		// apply gate on one qubit
		for (int i = 0; i < coordsLength; ++i) { 
			if (currentCoords->entry(i, 0) != (complex_t)0) {
				int qb = (i >> index) & 1;

				col = cols[qb];

				nextCoords->entry(i, 0) += currentCoords->entry(i, 0) * col->entry(qb, 0);
				nextCoords->entry(i ^ (1 << index), 0) += currentCoords->entry(i, 0) * col->entry(1 - qb, 0);
			}
		}

	}

	if (currentCoords == newCoords) {
		free(currentCoords);
	}

	coords = nextCoords;
}

void Quregister::applyGates(vector<pair<Matrix2, int>> gatesIndices) {
	Matrix2* newCoords = new Matrix2(coordsLength, 1);

	Matrix2* coordsArr[2] = { coords, newCoords };

	Matrix2* currentCoords = newCoords, * nextCoords = coords;

	int amountGates = gatesIndices.size();

	for (int i = 0; i < amountGates; ++i) {
		currentCoords = coordsArr[i % 2];
		nextCoords = coordsArr[(i + 1) % 2];
		nextCoords->zero();

		Matrix2& gate = gatesIndices[i].first;
		int index = gatesIndices[i].second;

		Matrix2* cols[2] = { &gate.col(0), &gate.col(1) };
		Matrix2* col;

		// apply gate on one qubit
		for (int i = 0; i < coordsLength; ++i) {
			if (currentCoords->entry(i, 0) != (complex_t)0) {
				int qb = (i >> index) & 1;

				col = cols[qb];

				nextCoords->entry(i, 0) += currentCoords->entry(i, 0) * col->entry(qb, 0);
				nextCoords->entry(i ^ (1 << index), 0) += currentCoords->entry(i, 0) * col->entry(1 - qb, 0);
			}
		}

	}

	if (currentCoords == newCoords) {
		free(currentCoords);
	}

	coords = nextCoords;
}


void Quregister::applyGateOnQubits(Matrix2& gate, int beginIndex, int endIndex) {
	Matrix2* newCoords = new Matrix2(coordsLength, 1);

	Matrix2* coordsArr[2] = { coords, newCoords };

	Matrix2* currentCoords = newCoords, * nextCoords = coords;


	for (int index = beginIndex; index < endIndex; ++index) {
		currentCoords = coordsArr[(index - beginIndex) % 2];
		nextCoords = coordsArr[(index - beginIndex + 1) % 2];
		nextCoords->zero();

		Matrix2* cols[2] = { &gate.col(0), &gate.col(1) };
		Matrix2* col;

		// apply gate on one qubit
		for (int i = 0; i < coordsLength; ++i) {
			if (currentCoords->entry(i, 0) != (complex_t)0) {
				int qb = (i >> index) & 1;

				col = cols[qb];

				nextCoords->entry(i, 0) += currentCoords->entry(i, 0) * col->entry(qb, 0);
				nextCoords->entry(i ^ (1 << index), 0) += currentCoords->entry(i, 0) * col->entry(1 - qb, 0);
			}
		}

	}

	if (currentCoords == newCoords) {
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

int Quregister::regMeasureInSubSpaces(vector<vector<Quregister>> bases) {
	return measureInSubSpaces(*this, bases);
}

int Quregister::regMeasureComputational() {
	return measureComputational(*this);
}

int Quregister::regMeasureComputational(int i, int j) {
	return measureComputational(*this, i, j);
}

complex_t rootOfUnityPower(int order, int power) {
	double angle = 2 * M_PI / order * (power % order);
	return complex_t(cos(angle), sin(angle));
}

Quregister& Quregister::QFT(Quregister& reg) {
	Quregister* retReg = new Quregister(reg.length, 0);

	complex_t sum = 0;

	for (int k = 0; k < reg.coordsLength; ++k) {
		sum = 0;

		for (int n = 0; n < reg.coordsLength; ++n) {
			sum += reg.getCoords()->entry(n, 0) * rootOfUnityPower(reg.coordsLength, n * k);
		}

		retReg->getCoords()->entry(k, 0) = 1 / sqrt(reg.coordsLength) * sum;
	}

	return *retReg;
}