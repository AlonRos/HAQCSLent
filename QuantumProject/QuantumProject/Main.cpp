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
#include "DeutschAlgorithm.h"

using namespace std;

int main() {
	Quregister q(2, 0);

	q.applyGateOnQubits(hadamard, 0, 2);

	q.getCoords()->print();

	cout << "\n" << q.regMeasureInSubSpaces({{Quregister(2, 0), Quregister(2, 2)}, {Quregister(2, 1), Quregister(2, 3)}}) << "\n";
	q.getCoords()->print();


}