#include <iostream>
#include "Matrix.h"
#include "Gates.h"
#include <random>
#include <chrono>
 
 using namespace std;
 
 int main() {
	 srand(time(NULL));

	 int c = 20;

	 int duration = 0;
	 for (int i = 0; i < c; ++i) {
		 Matrix& m1 = Matrix::randomMatrix(1000, 1000, 25);
		 //m1.print();
		 //cout << "\n";

		 Matrix& m2 = Matrix::randomMatrix(1000, 1000, 25);
		 //m2.print();
		 //cout << "\n";


		 auto start = chrono::high_resolution_clock::now();
		 m1 + m2;
		 auto stop = chrono::high_resolution_clock::now();
		 duration += duration_cast<chrono::milliseconds>(stop - start).count();
		 //cout << "\n";
		 //cout << "\n";


	 }
	 cout << duration / c;
 }