#ifndef Quregister_H
#define Quregister_H

#include <complex>
typedef std::complex<double> complex_t;


class Quregister {
private:
	int registerLength;
	complex_t* coords;

public:
	Quregister(int length, int num);


};







#endif