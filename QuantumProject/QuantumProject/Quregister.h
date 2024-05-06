#ifndef Quregister_H
#define Quregister_H

#include <complex>
#include "Matrix2.h"
#include <vector>

typedef std::complex<double> complex_t;


class Quregister {
private:
	int length;
	Matrix2* coords;
	int coordsLength;
	

public:
	// Initializing a Quantum Register to the value |num >
	Quregister(int length, int num);
	
	// apply gate to the whole register
	void applyGate(Matrix2& gate);

	// passing qubit in index i (from right)
	void applyGateOnQubit(Matrix2& gate, int i);

	// passing qubits from index i to index j (not including j)
	void applyGates(Matrix2* gates, int i, int j);

	void applyGates(vector<pair<Matrix2, int>> gatesIndices);

	// passing qubits from index i to index j (not including j) into the same gate
	void applyGateOnQubits(Matrix2& gate, int i, int j);

	void applyGateOnSubReg(Matrix2& gate, int i, int j);

	int regMeasure(vector<Quregister> basis);

	int regMeasureInSubSpaces(vector<vector<Quregister>> bases);

	int regMeasureComputational();

	// measure the part between qubit i and qubit j (not including j)
	int regMeasureComputational(int i, int j);

	int getRegLength();

	int getCoordsLength();

	Matrix2*& getCoords();

	static Quregister& QFT(Quregister& reg);
};

#endif