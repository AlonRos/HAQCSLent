#include "Quregister.h"

Quregister::Quregister(int length, int num) : registerLength(length) {
	int coordsLength = 2 << length;
	coords = new complex_t[coordsLength];
}