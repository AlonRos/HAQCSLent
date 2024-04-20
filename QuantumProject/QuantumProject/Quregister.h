#ifndef Quregister_H
#define Quregister_H

#include <complex>
#include "Matrix.h"
#include <vector>

typedef std::complex<double> complex_t;


class Quregister {
private:
	int length;
	Matrix* coords;
	int coordsLength;
	

public:
	Quregister(int length, int num);

	// Pass the qubits from i to j ([i,j)) through the gate
	void applyGate(int i, int j, Matrix& gate);
	
	void applyGate(Matrix& gate);

	int regMeasure(vector<Quregister> basis);

	int regMeasureComputational();

	int getRegLength();

	int getCoordsLength();

	Matrix* getCoords();

};







#endif