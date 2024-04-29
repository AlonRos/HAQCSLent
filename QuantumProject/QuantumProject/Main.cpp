#include <iostream>
#include "Log.h"
#include "Matrix2.h"
#include "Gates.h"
#include <random>
#include <chrono>
#include "Quregister.h"

#ifdef USEGPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CudaHeader.cuh"
#endif

#include "NegatingXFunction.h"

using namespace std;

int main() {
	int num = 3;
	int length = 13;

	Quregister reg(length, num);

	Quregister::QFT(reg).getCoords();

}