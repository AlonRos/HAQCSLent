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
#include "Grover'sAlgorithm.h"

#include <fstream>

using namespace std;

#define DEBUG3

#ifdef DEBUG
#define INPUT_TO_ALG "../../GUI/input_to_alg.txt"
#define OUTPUT_FROM_ALG "../../GUI/output_from_alg.txt"

#else
#define INPUT_TO_ALG "./input_to_alg.txt"
#define OUTPUT_FROM_ALG "./output_from_alg.txt"

#endif


int main() {
	int t;

	ofstream output(OUTPUT_FROM_ALG);
	ifstream input;

	output << "result\n";
	output.close();

	int groverTableW = 32, groverTableH = 30;

	int amountOnes, x, y;

	int groverN = groverTableW * groverTableH;
	int* f = new int[groverN];

	while (true) {
		cin >> t;

		output.open(OUTPUT_FROM_ALG);
		input.open(INPUT_TO_ALG);

		if (t == 0) { // terminate
			break;
		}

		else if (t == 1) { // grover
			for (int i = 0; i < groverN; ++i) {
				f[i] = 0;
			}

			input >> amountOnes;

			for (int i = 0; i < amountOnes; ++i) {
				input >> x >> y;
				
				f[y * groverTableW + x] = 1;
			}

			output << "result\n" << grover(f, groverN);
		}

		else if (t == 2) { // deutsch
			
		}

		output.close();
		input.close();
	}

	output.close();
	input.close();
	delete[] f;

}