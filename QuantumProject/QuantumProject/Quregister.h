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
	Quregister(int length, int num);
	
	void applyGate(Matrix2& gate);

	// passing qubit in index i (from right)
	void applyGateOnQubit(Matrix2& gate, int i);

	// passing qubits from index i to index j (not including j)
	void applyGates(Matrix2* gates, int i, int j);

	// passing qubits from index i to index j (not including j)
	void applyGateOnQubits(Matrix2& gate, int i, int j);

	int regMeasure(vector<Quregister> basis);

	int regMeasureComputational();

	int getRegLength();

	int getCoordsLength();

	Matrix2*& getCoords();

	static Quregister& QFT(Quregister& reg);
};

#endif