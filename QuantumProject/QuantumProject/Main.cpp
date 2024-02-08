#include <iostream>
#include "Matrix.h"

using namespace std;

int main() {
	Matrix m(3, 4, true);

	Matrix::mult(m, m);

	Matrix secondRow = m.row(2);

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 4; ++j) {
			m.entry(i, j) = 4 * i + j;
		}
	}

	for (int i = 0; i < 4; ++i) {
		cout << secondRow.entry(i, 0) << " ";
	}
	cout << "\n\n";
	
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 4; ++j) {
			cout << m.entry(i, j) << " ";
		}
		cout << "\n";
	}
}