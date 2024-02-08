#ifndef GPU_H
#define GPU_H

#include <cstdlib>

inline void* libraryMalloc(size_t size) {
#ifdef USEGPU
	return gpuMalloc(size);
#else
	return malloc(size);
#endif

}
void* gpuMalloc(size_t size);

inline void* libraryCalloc(size_t size) {
#ifdef USEGPU
	return gpuCalloc(size);
#else
	return calloc(size, 1);
#endif

}

void* gpuCalloc(size_t size);


#endif