#ifndef Utils_H
#define Utils_H

#include <iterator>

#define Exception(ex, message, ...) ex(format(message, __VA_ARGS__))

using namespace std;

void copyArr(void* src, void* dest, int length, int sizeOfElement);




#endif