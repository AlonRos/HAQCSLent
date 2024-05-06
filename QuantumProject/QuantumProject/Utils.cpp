#include "Utils.h"
#include <random>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0, 1);

void copyArr(void* src, void* dest, int length, int sizeOfElement) {
	 copy((char*) src, (char*)src + length * sizeOfElement, (char*)dest);
}

double complexNormSquared(complex_t z) {
	double real = z.real(), imag = z.imag();
	return real * real + imag * imag;
}

double rand01() {
	return dis(gen);
}

int randBound(int bound) {
	return (int)(rand01() * bound);
}
