#ifndef Qubit_H
#define Qubit_H

#include "complex"
#include "MeasurementOperator.h"

using namespace std;

class Qubit {
private:
	complex<double> a, b; // then the qubit is a|0> + b|1>


public:
	Qubit();

	complex<double> measure(MeasurementOperator M);



};





#endif