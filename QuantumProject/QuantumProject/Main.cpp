#include <iostream>
#include "Matrix.h"

using namespace std;

int main() {
	Matrix m(4, 4, true);

	Matrix::mult(m, m);

	Matrix secondRow = m.row(2);

	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			m.entry(i, j) = 4 * i + j;
		}
	}

	for (int i = 0; i < 4; ++i) {
		cout << secondRow.entry(0, i) << " ";
	}
	cout << "\n\n";
	

	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			cout << m.entry(i, j) << " ";
		}
		cout << "\n";
	}
	cout << "\n";


	Matrix x = secondRow * m;

	for (int i = 0; i < 1; ++i) {
		for (int j = 0; j < 4; ++j) {
			cout << x.entry(i, j) << " ";
		}
		cout << "\n";
	}
}