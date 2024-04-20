#include "Measurement.h"

double rand01() {
	return ((double)rand()) / RAND_MAX;
}

int measure(Quregister reg, vector<Quregister> basis) {
	int coordsLength = reg.getCoordsLength();

	Matrix** measurementMatrices = (Matrix**) operator new (sizeof(Matrix*) * coordsLength);
	Matrix* currentCoords;

	for (int i = 0; i < coordsLength; ++i) {
		currentCoords = basis[i].getCoords();
		measurementMatrices[i] = &Matrix::mult(*currentCoords, currentCoords->conjTranspose());
	}

	double* probs = new double[coordsLength];
	currentCoords = reg.getCoords();
	for (int i = 0; i < coordsLength; ++i) {
		probs[i] = (*measurementMatrices[i] * *currentCoords).normSquared();

		free(measurementMatrices[i]);
	}

	double randNumber = rand01();

	for (int i = 0; i < coordsLength; ++i) {
		if (randNumber < probs[i]) return i;
		randNumber -= probs[i];
	}

	
	return 0;
}

int measureComputational(Quregister reg) {
	vector<Quregister> basis;
	int regLength = reg.getRegLength(), coordsLength = reg.getCoordsLength();

	for (int i = 0; i < coordsLength; ++i) {
		basis.push_back(Quregister(regLength, i));
	}

	return measure(reg, basis);
}