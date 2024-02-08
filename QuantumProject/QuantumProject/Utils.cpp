#include "Utils.h"

void copyArr(char* src, char* dest, int length, int sizeOfElement) {
	 copy(src, src + length * sizeOfElement, dest);
}