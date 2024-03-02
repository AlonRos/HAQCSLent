#include "Matrix.h"
#include "GPU.h"
#include "Utils.h"
#include <iostream>
#include <random>
#include <chrono>

using namespace std;

Matrix::Matrix(int m, int n, bool JAisRow) : m(m), n(n), jumpArrayIsRow(JAisRow) {
	complex_t* allocatedMemory = (complex_t*)libraryCalloc(m * n * sizeof(complex_t));
	if (JAisRow) {
		elements = (JumpArray<complex_t>*) operator new (m * sizeof(JumpArray<complex_t>));

		for (int i = 0; i < m; ++i) {
			elements[i] = JumpArray<complex_t>(allocatedMemory + i, m * sizeof(complex_t), n);
		}
	}

	else {
		elements = (JumpArray<complex_t>*) operator new (n * sizeof(JumpArray<complex_t>));

		for (int i = 0; i < n; ++i) {
			elements[i] = JumpArray<complex_t>(allocatedMemory + i, n * sizeof(complex_t), m);
		}
	}

}

Matrix::Matrix(int m, int n) :Matrix(m, n, m < n) {}


// Constructor which gets an array of JA and copys them. The elements themselves does'nt get copied
Matrix::Matrix(int m, int n, bool JAisRow, JumpArray<complex_t> elements[]) : m(m), n(n), jumpArrayIsRow(JAisRow), elements(elements) {}

complex_t& Matrix::entry(int rowIndex, int colIndex) {
	if (!(0 <= rowIndex && rowIndex < m && 0 <= colIndex && colIndex < n)) {
		throw Exception(out_of_range, "Tried accessing matrix with dim {} x {} at entry ({}, {})", m, n, rowIndex, colIndex);
	}
	if (jumpArrayIsRow) {
		return elements[rowIndex][colIndex];
	}
	else {
		return elements[colIndex][rowIndex];

	}
}

Matrix& Matrix::rows(int i, int j) {
	if (!(0 <= i && i < m && 0 <= j && j <= m)) {
		throw Exception(out_of_range, "Tried accessing matrix with dim {} x {} with rows ({}, {})", m, n, i, j);
	}

	Matrix* returnRow;
	JumpArray<complex_t>* rowElements = nullptr;

	if (jumpArrayIsRow) { // if m < n then j - i < n
		rowElements = (JumpArray<complex_t>*) operator new ((j - i) * sizeof(JumpArray<complex_t>));

		copyArr(&elements[i], rowElements, j - i, sizeof(JumpArray<complex_t>));

		returnRow = new Matrix(j - i, n, true, rowElements);
	}
	else {
		if (j - i < n) { // new matrix is rowish
			rowElements = (JumpArray<complex_t>*) operator new ((j - i) * sizeof(JumpArray<complex_t>));

			for (int k = 0; k < j - i; ++k) {
				rowElements[k] = *new JumpArray<complex_t>(&entry(k + i, 0), sizeof(complex_t), n);
			}
			returnRow = new Matrix(j - i, n, true, rowElements);
		}
		else { // new matrix is colish
			rowElements = (JumpArray<complex_t>*) operator new (n * sizeof(JumpArray<complex_t>));

			for (int k = 0; k < n; ++k) {
				rowElements[k] = *new JumpArray<complex_t>(&entry(i, k), n * sizeof(complex_t), j - i);
			}
			returnRow = new Matrix(j - i, n, false, rowElements);
		}
	}

	return *returnRow;
}

Matrix& Matrix::row(int rowIndex) {
	return rows(rowIndex, rowIndex + 1);
}

Matrix& Matrix::col(int colIndex) {
	Matrix* returnCol;
	if (jumpArrayIsRow) {
		JumpArray<complex_t> colJA = JumpArray<complex_t>(&entry(0, colIndex), sizeof(complex_t), m);
		JumpArray<complex_t> colElements[] = { colJA };
		returnCol = new Matrix(m, 1, false, colElements);
	}
	else {
		JumpArray<complex_t> colElements[] = { elements[colIndex] };
		returnCol = new Matrix(m, 1, false, colElements);
	}

	return *returnCol;

}


Matrix& Matrix::transpose() {
	Matrix* returnMatrix = new Matrix(n, m, !jumpArrayIsRow, elements);
	return *returnMatrix;
}


Matrix& Matrix::cpuMult(Matrix& A, Matrix& B) {
	if (A.n != B.m) {
		throw Exception(runtime_error, "Cannot multiply a {} x {} matrix with a {} x {} matrix", A.m, A.n, B.m, B.n);
	}

	Matrix* returnMatrix = new Matrix(A.m, B.n);
	complex_t res = 0;

	for (int i = 0; i < A.m; ++i) {
		for (int j = 0; j < B.n; ++j) {

			for (int k = 0; k < A.n; ++k) {
				res += A.entry(i, k) * B.entry(k, j);
			}

			returnMatrix->entry(i, j) = res;
			
			res = 0;
		}
	}

	return *returnMatrix;
	
}

Matrix& Matrix::operator*(Matrix& other) {
	return Matrix::mult(*this, other);
}

Matrix& Matrix::operator*(complex_t scalar) {
	Matrix* returnMatrix = new Matrix(m, n);

	complex_t* returnMatrixFirstElement = &returnMatrix->entry(0, 0);
	complex_t* firstElement = &entry(0, 0);

	for (int i = 0; i < m * n; ++i) {
		returnMatrixFirstElement[i] = firstElement[i] * scalar;
	}

	return *returnMatrix;
}

Matrix& Matrix::fromArray(int m, int n, bool JAisRow, complex_t* arr) {
	Matrix* returnMatrix = new Matrix(m, n, JAisRow);

	if (JAisRow) {
		complex_t* firstElement = &returnMatrix->entry(0, 0);
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				firstElement[i + m * j] = arr[j + n * i];
			}
		}
	}
	else {
		copy(arr, arr + m * n, &returnMatrix->entry(0, 0));
	}

	return *returnMatrix;
}

Matrix& Matrix::fromArray(int m, int n, complex_t* arr) {
	return fromArray(m, n, n > m, arr);
}

void Matrix::print() {
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			cout << entry(i, j) << " ";
		}
		cout << '\n';
	}
}

Matrix& Matrix::randomMatrix(int m, int n) {
	Matrix* returnMatrix = new Matrix(m, n);

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			returnMatrix->entry(i, j) = rand();
		}
	}
	return *returnMatrix;
}

Matrix& Matrix::randomMatrix(int m, int n, int bound) {
	srand(time(NULL));

	Matrix* returnMatrix = new Matrix(m, n);

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			returnMatrix->entry(i, j) = rand() % bound;
		}
	}
	return *returnMatrix;

}