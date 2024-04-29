#include "Matrix2.h"
#include <algorithm>
#include <iostream>

using namespace std;

Matrix2::Matrix2(int m, int n) : m(m), n(n), rowwise(true), jump(n) {
	elements = (complex_t*) calloc(m * n * sizeof(complex_t), 1);
}

Matrix2::Matrix2(int m, int n, bool rowwise) : m(m), n(n), rowwise(rowwise) {
	jump = rowwise ? n : m;

	elements = (complex_t*) malloc(m * n * sizeof(complex_t));
}

Matrix2::Matrix2(int m, int n, complex_t* elements, bool rowwise, int jump) : m(m), n(n), elements(elements), rowwise(rowwise), jump(jump), toFree(false) {}

Matrix2::Matrix2(int m, int n, complex_t* arr) : m(m), n(n), rowwise(true), jump(n) {
	elements = (complex_t*) malloc(m * n * sizeof(complex_t));
	copy(arr, arr + m * n, elements);
}

Matrix2::~Matrix2() {
	if (toFree) {
		free(elements);
	}
}

complex_t& Matrix2::entry(int rowIndex, int colIndex) {
	if (rowwise) {
		return elements[rowIndex * jump + colIndex];
	}
	else {
		return elements[colIndex * jump + rowIndex];
	}
}

Matrix2& Matrix2::rows(int i, int j) {
	complex_t* rowsElements = &entry(i, 0);
	return *new Matrix2(j - i, n, rowsElements, rowwise, jump);
}

Matrix2& Matrix2::cols(int i, int j) {
	complex_t* colsElements = &entry(0, i);
	return *new Matrix2(m, j - i, colsElements, rowwise, jump);
}

void Matrix2::zero() {
	if (rowwise) {
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				entry(i, j) = 0;
			}
		}
	}

	else {
		for (int j = 0; j < n; ++j) {
			for (int i = 0; i < m; ++i) {
				entry(i, j) = 0;
			}
		}
	}
}

Matrix2& Matrix2::transpose() {
	return *new Matrix2(n, m, elements, !rowwise, jump);
}

Matrix2& Matrix2::conjTranspose() {
	Matrix2* retMatrix = new Matrix2(n, m, !rowwise);

	int newJump;
	if (rowwise) {
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				retMatrix->entry(j, i) = conj(entry(i, j));
			}
		}
		newJump = n;
	}

	else {
		for (int j = 0; j < n; ++j) {
			for (int i = 0; i < m; ++i) {
				retMatrix->entry(j, i) = conj(entry(i, j));
			}
		}
		newJump = m;
	}

	return *retMatrix;
}

double Matrix2::normSquared() {
	double res = 0;

	if (rowwise) {
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				res += complexNormSquared(entry(i, j));
			}
		}
	}
	else {
		for (int j = 0; j < n; ++j) {
			for (int i = 0; i < m; ++i) {
				res += complexNormSquared(entry(i, j));
			}
		}
	}

	return res;
}

void Matrix2::cpuAddIn(Matrix2& A, Matrix2& B, Matrix2& saveIn) {
	if (A.m != B.m or A.n != B.n) {
		throw Exception(runtime_error, "Cannot multiply a {} x {} matrix with a {} x {} matrix", A.m, A.n, B.m, B.n);
	}

	int m = A.m, n = A.n;

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			saveIn.entry(i, j) = A.entry(i, j) + B.entry(i, j);
		}
	}
}

void Matrix2::cpuMultIn(Matrix2& A, Matrix2& B, Matrix2& saveIn) {
	if (A.n != B.m) {
		throw Exception(runtime_error, "Cannot multiply a {} x {} matrix with a {} x {} matrix", A.m, A.n, B.m, B.n);
	}

	complex_t* res = new complex_t[A.m * B.n];

	for (int i = 0; i < A.m; ++i) {
		for (int k = 0; k < A.n; ++k) {
			for (int j = 0; j < B.n; ++j) {
				res[B.n * i + j] += A.entry(i, k) * B.entry(k, j);
			}
		}
	}

	for (int i = 0; i < A.m; ++i) {
		for (int j = 0; j < B.n; ++j) {
			saveIn.entry(i, j) = res[B.n * i + j];
		}
	}
}

void Matrix2::cpuKroneckerIn(Matrix2& A, Matrix2& B, Matrix2& saveIn) {
	for (int i = 0; i < A.m; ++i) {
		for (int j = 0; j < A.n; ++j) {
			for (int k = 0; k < B.m; ++k) {
				for (int l = 0; l < B.n; ++l) {
					saveIn.entry(i * B.m + k, j * B.n + l) = A.entry(i, j) * A.entry(k, l);

				}
			}
		}
	}
}

Matrix2& Matrix2::operator*(complex_t scalar) {
	Matrix2* retMatrix = new Matrix2(m, n, rowwise);

	int newJump;

	if (rowwise) {
		for (int i = 0; i < m; ++i) {
			for (int j = 0; j < n; ++j) {
				retMatrix->entry(i, j) = entry(i, j) * scalar;
			}
		}
		newJump = n;
	}

	else {
		for (int j = 0; j < n; ++j) {
			for (int i = 0; i < m; ++i) {
				retMatrix->entry(i, j) = entry(i, j) * scalar;
			}
		}
		newJump = m;
	}

	return *retMatrix;
}

Matrix2& Matrix2::randomMatrix(int m, int n, int bound) {
	Matrix2* returnMatrix = new Matrix2(m, n, true);

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			returnMatrix->entry(i, j) = complex_t(randBound(bound), randBound(bound));
		}
	}

	return *returnMatrix;

}

void Matrix2::print() {
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (entry(i, j).real() == 0 && entry(i, j).imag() != 0) {
				cout << entry(i, j).imag() << "i" << " ";

			}
			else if (entry(i, j).imag() > 0) {
				cout << entry(i, j).real() << "+" << entry(i, j).imag() << "i" << " ";
			}
			else if (entry(i, j).imag() < 0) {
				cout << entry(i, j).real() << entry(i, j).imag() << "i" << " ";
			}
			else {
				cout << entry(i, j).real() << " ";

			}
		}
		cout << '\n';
	}
}