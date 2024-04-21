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

	// Pass the qubits from i to j ([i,j)) through the gate
	void applyGate(int i, int j, Matrix2& gate);
	
	void applyGate(Matrix2& gate);

	int regMeasure(vector<Quregister> basis);

	int regMeasureComputational();

	int getRegLength();

	int getCoordsLength();

	Matrix2*& getCoords();

};







#endif