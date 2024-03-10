#ifndef JumpArray_H
#define JumpArray_H

#include "Log.h"
#include "Utils.h"
#include <format>

using namespace std;

template<typename type>
class JumpArray {
private:
	type* firstPointer;
	int jump;
	int length;


public:
	JumpArray(type* ptr, int jump, int length);

	inline type& operator[](int index);

	type* getPtr();

};

template<typename type>
JumpArray<type>::JumpArray(type* ptr, int jump, int length) : jump(jump), firstPointer(ptr), length(length) {}

template<typename type>
inline type& JumpArray<type>::operator[](int index) {
	if (index >= length) {
		throw Exception(out_of_range, "Tried accessing jumpArray with length {} at index {}", length, index);
	}
	return *(type*)((char*)firstPointer + index * jump);
}

template<typename type>
type* JumpArray<type>::getPtr() {
	return firstPointer;
}



#endif
