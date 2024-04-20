#ifndef Utils_H
#define Utils_H

#include <iterator>
#include <complex>

typedef std::complex<double> complex_t;


#define Exception(ex, message, ...) ex(format(message, __VA_ARGS__))

using namespace std;

void copyArr(void* src, void* dest, int length, int sizeOfElement);

double complexNormSquared(complex_t z);

double rand01();

#endif