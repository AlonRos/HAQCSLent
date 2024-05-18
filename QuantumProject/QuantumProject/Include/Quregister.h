#ifndef Quregister_H
#define Quregister_H

#include <complex>
#include "../Include/Matrix2.h"
#include <vector>

typedef std::complex<double> complex_t;

struct gateIndexSize {
	Matrix2& gate;
	int beginIndex;
	int endIndex;
};

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

	// apply gates on the register
	void applyGates(vector<gateIndexSize> gatesIndicesSizes);

	// passing qubits from index i to index j (not including j) into the same gate
	void applyGateOnQubits(Matrix2& gate, int i, int j);

	// apply gate on part of the register 
	void applyGateOnSubReg(Matrix2& gate, int i, int j);

	// measure the register with respect to an orthonormal basis
	int regMeasure(vector<Quregister> basis);

	// mesasure the register with respect to sub spaces
	int regMeasureInSubSpaces(vector<vector<Quregister>> bases);

	// measure the register in computational basis
	int regMeasureComputational();

	// measure the part between qubit i and qubit j (not including j)
	int regMeasureComputational(int i, int j);

	int getRegLength();

	int getCoordsLength();

	Matrix2*& getCoords();

	static Quregister& QFT(Quregister& reg);
};

#endif