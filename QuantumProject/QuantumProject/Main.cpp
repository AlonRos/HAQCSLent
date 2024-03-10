#include <iostream>
#include "Matrix.h"
#include "Gates.h"

using namespace std;

int main() {
	
	
	Matrix& m = Matrix::randomMatrix(7, 9, 25);
	m.print();

	cout << '\n';

	m.transpose().rows(1, 5).transpose().rows(2, 3).print();
		

}