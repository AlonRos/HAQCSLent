#include "Utils.h"

void copyArr(void* src, void* dest, int length, int sizeOfElement) {
	 copy((char*) src, (char*)src + length * sizeOfElement, (char*)dest);
}