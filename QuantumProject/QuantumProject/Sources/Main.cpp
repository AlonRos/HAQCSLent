#include <iostream>
#include "../Include/Log.h"
#include "../Include/Matrix2.h"
#include "../Include/Gates.h"
#include <random>
#include <chrono>
#include "../Include/Quregister.h"

#ifdef USEGPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../CUDA/CudaHeader.cuh"
#endif

#include "../Include/NegatingXFunction.h"
#include "../Include/DeutschAlgorithm.h"
#include "../Include/Grover'sAlgorithm.h"

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
#ifdef USEGPU
	init();
#endif
	int t;

	ofstream output(OUTPUT_FROM_ALG);
	ifstream input(INPUT_TO_ALG);

	int groverTableW, groverTableH, deutschTableW, deutschN;

	cin >> groverTableW >> groverTableH >> deutschTableW >> deutschN;
	output << "result\n";

	input.close();
	output.close();

	int amountOnes, x, y;

	int groverN = groverTableW * groverTableH;
	int* groverF = new int[groverN];

	int deutschSize = 1 << deutschN;
	int* deutschF = new int[deutschSize];

	while (true) {
		cin >> t; // which algorithm to run

		output.open(OUTPUT_FROM_ALG);
		input.open(INPUT_TO_ALG);

		if (t == 0) { // terminate
			break;
		}

		else if (t == 1) { // grover
			std::fill(groverF, groverF + groverN, 0);

			input >> amountOnes;

			// set the function to the wanted one
			for (int i = 0; i < amountOnes; ++i) {
				input >> x >> y;
				
				groverF[y * groverTableW + x] = 1;
			}

			output << "result\n" << grover(groverF, groverN);  // write the result to the file
		}

		else if (t == 2) { // deutsch
			input >> x;

			// set the function to the wanted one
			if (x == 0 || x == 1) {
				std::fill(deutschF, deutschF + deutschSize, x);
			}

			else if (x == 2) { // balanced
				std::fill(deutschF, deutschF + deutschSize, 0);

				// set the function to the wanted one
				for (int i = 0; i < deutschSize / 2; ++i) {
					input >> x >> y;

					deutschF[y * deutschTableW + x] = 1;
				}
			}

			output << "result\n" << isBalanced(deutschF, deutschN); // write the result to the file
		}

		output.close();
		input.close();
	}

	output.close();
	input.close();
	delete[] groverF;

}