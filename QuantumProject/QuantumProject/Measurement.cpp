#include "Measurement.h"

double rand01() {
	return ((double)rand()) / RAND_MAX;
}

int randomIndexProbs(double* probs, int probesLength) {
	double randNumber = rand01();

	for (int i = 0; i < coordsLength; ++i) {
		if (randNumber < probs[i]) {
			return i;
		}
		randNumber -= probs[i];
	}


	return 0;
}

int measure(Quregister& reg, vector<Quregister> basis) {
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

	int i = randomIndexProbs(probs, coordsLength);
	reg.getCoords() = basis[i].getCoords();
	return i;


}

int measureComputational(Quregister& reg) {
	vector<Quregister> basis;
	int regLength = reg.getRegLength(), coordsLength = reg.getCoordsLength();

	Matrix* coords = reg.getCoords();
	double* probs = new double[coordsLength];
	for (int i = 0; i < coordsLength; ++i) {
		probs[i] = complexNormSquared(coords.entry(i, 0));
	}

	int i = randomIndexProbs(probs, coordsLength);
	reg.getCoords() = basis[i].getCoords();
	return i;
}