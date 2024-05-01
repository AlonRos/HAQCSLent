#include "Measurement.h"
#include <vector>

using namespace std;

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


// otrhonormal basis
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

	int chosenIndex = randomIndexProbs(probs, coordsLength);
	delete[] probs;

	Matrix2::multIn(*measurementMatrices[chosenIndex], *reg.getCoords(), *reg.getCoords());

	return chosenIndex;
}

// each basis is an orhtonormal basis of V_a such that H^n = V_1 \directsum ... \dirrectsum V_k
int measureInSubSpaces(Quregister& reg, vector<vector<Quregister>> bases) {
	int coordsLength = reg.getCoordsLength(), amountSpaces = bases.size();

	Matrix2** measurementMatrices = (Matrix2**) operator new (sizeof(Matrix2*) * amountSpaces);
	Matrix2* currentCoords;

	int currDim;
	vector<Quregister> currBase;
	Matrix2* currMat;

	for (int a = 0; a < amountSpaces; ++a) {
		measurementMatrices[a] = new Matrix2(coordsLength, coordsLength);
		currBase = bases[a];
		currDim = currBase.size();
		
		for (int i = 0; i < currDim; ++i) {
			currentCoords = currBase[i].getCoords();
			currMat = &Matrix2::mult(*currentCoords, currentCoords->conjTranspose());
			Matrix2::addIn(*measurementMatrices[i], *currMat, *measurementMatrices[i]);
		}
	}

	double* probs = new double[amountSpaces];
	currentCoords = reg.getCoords();
	for (int i = 0; i < amountSpaces; ++i) {
		probs[i] = (*measurementMatrices[i] * *currentCoords).normSquared();

		free(measurementMatrices[i]);
	}

	int chosenSpaceIndex = randomIndexProbs(probs, amountSpaces);
	delete[] probs;

	Matrix2::multIn(*measurementMatrices[chosenSpaceIndex], *reg.getCoords(), *reg.getCoords());

	return chosenSpaceIndex;
}

int measureComputational(Quregister& reg) {
	int regLength = reg.getRegLength(), coordsLength = reg.getCoordsLength();

	Matrix2* coords = reg.getCoords();
	double* probs = new double[coordsLength];
	for (int i = 0; i < coordsLength; ++i) {
		probs[i] = complexNormSquared(coords->entry(i, 0));
	}

	int chosenIndex = randomIndexProbs(probs, coordsLength);
	delete[] probs;

	complex_t chosenIndexEntryNormalized = reg.getCoords()->entry(chosenIndex, 0) / abs(reg.getCoords()->entry(chosenIndex, 0));
	free(reg.getCoords());
	reg.getCoords() = new Matrix2(coordsLength, 1);
	reg.getCoords()->entry(chosenIndex, 0) = chosenIndexEntryNormalized;

	return chosenIndex;
}

int measureComputational(Quregister& reg, int beginIndex, int endIndex) {
	int regLength = reg.getRegLength(), coordsLength = reg.getCoordsLength();;

	int amountSpaces = 1 << (endIndex - beginIndex), amountInSpace = 1 << (regLength - endIndex + beginIndex);
	Matrix2* coords = reg.getCoords();
	double* probs = new double[amountSpaces];

	// space in index a is when the part from i to j equals a
	for (int a = 0; a < amountSpaces; ++a) {
		probs[a] = 0;

		for (int indexInSpace = 0; indexInSpace < amountInSpace; ++indexInSpace) {
			int indexInComputational = (indexInSpace & ~((1 << beginIndex) - 1)) | (a << beginIndex) | (indexInSpace & ((1 << beginIndex) - 1));
			probs[a] += complexNormSquared(coords->entry(indexInComputational, 0));
		}

	}

	int chosenSpaceIndex = randomIndexProbs(probs, amountSpaces);
	double normOfProjection = sqrt(probs[chosenSpaceIndex]);
	delete[] probs;

	Matrix2* newCoords = new Matrix2(coordsLength, 1);

	for (int indexInSpace = 0; indexInSpace < amountInSpace; ++indexInSpace) {
		int indexInComputational = (indexInSpace & ~((1 << beginIndex) - 1)) | (chosenSpaceIndex << beginIndex) | (indexInSpace & ((1 << beginIndex) - 1));
		newCoords->entry(indexInComputational, 0) = reg.getCoords()->entry(indexInComputational, 0) / normOfProjection;
	}

	free(reg.getCoords());

	reg.getCoords() = newCoords;

	return chosenSpaceIndex;
}