#ifndef Quregister_H
#define Quregister_H

#include <complex>
#include "Matrix.h"

typedef std::complex<double> complex_t;


class Quregister {
private:
	int length;
	Matrix* coords;
	

public:
	Quregister(int length, int num);

	// Pass the qubits from i to j ([i,j)) through the gate
	void applyGate(int i, int j, Matrix& gate);
	
	void applyGate(Matrix& gate);

	


};







#endif