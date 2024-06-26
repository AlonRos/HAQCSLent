#include "../Include/Measurement.h"
#include <vector>

using namespace std;

int randomIndexProbs(double* probs, int probsLength) {
	double randNumber = rand01(); // gives a random number in [0, 1)

	for (int i = 0; i < probsLength; ++i) {
		if (randNumber < probs[i]) {
			return i;
		}
		randNumber -= probs[i];
	}


	return probsLength - 1;
}

int randomIndexProjs(complex_t* projs, int projsLength) {
	double randNumber = rand01();

	double temp;
	for (int i = 0; i < projsLength; ++i) {
		temp = abs(projs[i]);
		temp = temp * temp;

		if (randNumber < temp) {
			return i;
		}
		randNumber -= temp;
	}


	return projsLength - 1;
}


// otrhonormal basis
int measure(Quregister& reg, vector<Quregister> basis) {
	int coordsLength = reg.getCoordsLength();

	complex_t* projs = new complex_t[coordsLength];

	Matrix2* innerProduct = new Matrix2(1, 1, true), *vStar;

	for (int i = 0; i < coordsLength; ++i) {
		vStar = &basis[i].getCoords()->conjTranspose();
		Matrix2::multIn(*vStar, *reg.getCoords(), *innerProduct); // calculate the inner product

		delete vStar;

		projs[i] = innerProduct->entry(0, 0); // the inner product (the projection)
	}

	delete innerProduct;

	int chosenIndex = randomIndexProjs(projs, coordsLength); // choose index according to projections

	// write the result and normalize
	Matrix2* chosenVector = basis[chosenIndex].getCoords();
	delete reg.getCoords();
	reg.getCoords() = &((*chosenVector) * (projs[chosenIndex] / abs(projs[chosenIndex])));

	delete[] projs;

	return chosenIndex;
}

// each basis is an orhtonormal basis of V_a such that H^n = V_1 \directsum ... \dirrectsum V_k
int measureInSubSpaces(Quregister& reg, vector<vector<Quregister>> bases) {
	int coordsLength = reg.getCoordsLength(), amountSpaces = bases.size();

	double* probs = new double[amountSpaces];

	int currDim;
	vector<Quregister> currBasis;

	Matrix2* innerProduct = new Matrix2(1, 1, true), * vStar;

	for (int a = 0; a < amountSpaces; ++a) { // loop through all the spaces
		currBasis = bases[a];
		currDim = currBasis.size();
		
		probs[a] = 0;

		for (int i = 0; i < currDim; ++i) { // loop through all the basis vectors of this sub space
			vStar = &currBasis[i].getCoords()->conjTranspose();
			Matrix2::multIn(*vStar, *reg.getCoords(), *innerProduct); // the inner product
			delete vStar;

			probs[a] += complexNormSquared(innerProduct->entry(0, 0)); // |proj|^2

		}
	}

	delete innerProduct;


	int chosenSpaceIndex = randomIndexProbs(probs, amountSpaces);
	delete[] probs;

	Matrix2* projectionMatrix = new Matrix2(coordsLength, coordsLength);
	int chosenSpaceDim = bases[chosenSpaceIndex].size();

	Matrix2* conjTranspose, *currMat;
	currBasis = bases[chosenSpaceIndex]; // the chosen basis

	for (int i = 0; i < chosenSpaceDim; ++i) {
		conjTranspose = &currBasis[i].getCoords()->conjTranspose(); // (currBasis[i])*

		currMat = &Matrix2::mult(*currBasis[i].getCoords(), *conjTranspose); // get the projection matrix on currBasis[i]
		Matrix2::addIn(*projectionMatrix, *currMat, *projectionMatrix); // add the curent projection matrix to the projection martix on the subspace

		delete conjTranspose;
		delete currMat;
	}

	Matrix2::multIn(*projectionMatrix, *reg.getCoords(), *reg.getCoords()); // projecting the state on the chosen sub space
	delete projectionMatrix;

	// normalizing the vector
	double regNorm = sqrt(reg.getCoords()->normSquared());
	for (int i = 0; i < coordsLength; ++i) {
		reg.getCoords()->entry(i, 0) /= regNorm;
	}

	return chosenSpaceIndex;
}

int measureComputational(Quregister& reg) {
	int regLength = reg.getRegLength(), coordsLength = reg.getCoordsLength();

	Matrix2* coords = reg.getCoords();
	double* probs = new double[coordsLength];
	for (int i = 0; i < coordsLength; ++i) {
		probs[i] = complexNormSquared(coords->entry(i, 0)); // calculate the probability to get | i >
	}

	int chosenIndex = randomIndexProbs(probs, coordsLength); // choose a value
	delete[] probs;

	// write the result and normalize
	complex_t chosenIndexEntryNormalized = reg.getCoords()->entry(chosenIndex, 0) / abs(reg.getCoords()->entry(chosenIndex, 0));
	delete reg.getCoords();
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
			int indexInComputational = ((indexInSpace & ~((1 << beginIndex) - 1)) << endIndex) | (a << beginIndex) | (indexInSpace & ((1 << beginIndex) - 1));
			probs[a] += complexNormSquared(coords->entry(indexInComputational, 0));
		}
	}

	int chosenSpaceIndex = randomIndexProbs(probs, amountSpaces);
	double normOfProjection = sqrt(probs[chosenSpaceIndex]);
	delete[] probs;

	Matrix2* newCoords = new Matrix2(coordsLength, 1);

	for (int indexInSpace = 0; indexInSpace < amountInSpace; ++indexInSpace) {
		int indexInComputational = ((indexInSpace & ~((1 << beginIndex) - 1)) << endIndex) | (chosenSpaceIndex << beginIndex) | (indexInSpace & ((1 << beginIndex) - 1));
		newCoords->entry(indexInComputational, 0) = reg.getCoords()->entry(indexInComputational, 0) / normOfProjection;
	}

	delete reg.getCoords();

	reg.getCoords() = newCoords;

	return chosenSpaceIndex;
}