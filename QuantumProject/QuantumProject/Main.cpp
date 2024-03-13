#include <iostream>
#include "Matrix.h"
#include "Gates.h"

using namespace std;

int main() {
	
	
	Matrix& m = Matrix::randomMatrix(7, 9, 25);
	m.print();

	cout << '\n';

	m.cols(1, 5).rows(2, 3).print();
	m.rows(2, 3).cols(1, 5).print();


}