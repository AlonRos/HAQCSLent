#include <iostream>
#include "Matrix.h"
#include "Gates.h"

using namespace std;

int main() {
	
	
	Matrix& m = Matrix::randomMatrix(5, 5, 25);
	m.print();

	cout << '\n';

	m.row(4).print();
		

}