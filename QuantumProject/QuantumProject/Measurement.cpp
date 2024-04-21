#include "Measurement.h"

int randomIndexProbs(double* probs, int probsLength) {
	double randNumber = rand01();

	for (int i = 0; i < probsLength; ++i) {
		if (randNumber < probs[i]) {
			return i;
		}
		randNumber -= probs[i];
	}


	return probsLength - 1;
}

int measure(Quregister& reg, vector<Quregister> basis) {
	int coordsLength = reg.getCoordsLength();

	Matrix2** measurementMatrices = (Matrix2**) operator new (sizeof(Matrix2*) * coordsLength);
	Matrix2* currentCoords;

	for (int i = 0; i < coordsLength; ++i) {
		currentCoords = basis[i].getCoords();
		measurementMatrices[i] = &Matrix2::mult(*currentCoords, currentCoords->conjTranspose());
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
	int regLength = reg.getRegLength(), coordsLength = reg.getCoordsLength();

	Matrix2* coords = reg.getCoords();
	double* probs = new double[coordsLength];
	for (int i = 0; i < coordsLength; ++i) {
		probs[i] = complexNormSquared(coords->entry(i, 0));
	}

	int i = randomIndexProbs(probs, coordsLength);
	reg.getCoords() = new Matrix2(coordsLength, 1);
	reg.getCoords()->entry(i, 0) = 1;

	return i;
}