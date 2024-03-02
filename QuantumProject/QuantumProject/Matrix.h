#ifndef Matrix_H
#define Matrix_H

#include <complex>
#include "JumpArray.h"

typedef std::complex<double> complex_t;


class Matrix { // m * n Matrix
private:
	JumpArray<complex_t>* elements; // Array of JumpArrays 
	bool jumpArrayIsRow;

	static Matrix& gpuMult(Matrix& A, Matrix& B);
	static Matrix& cpuMult(Matrix& A, Matrix& B);


public:
	int m, n;

	// Zeros Every Entry
	Matrix(int m, int n, bool JAisRow);

	Matrix(int m, int n);

	Matrix(int m, int n, bool JAisRow, JumpArray<complex_t> elements[]);

	static Matrix& fromArray(int m, int n, bool JAisRow, complex_t* arr);

	static Matrix& fromArray(int m, int n, complex_t* arr);

	inline static Matrix& mult(Matrix& A, Matrix& B);

	Matrix& operator*(Matrix& other);

	Matrix& operator*(complex_t scalar);

	complex_t& entry(int rowIndex, int colIndex);

	// Return a matrix containing the rows i to j (including i, not including j). The elements are in the same address as this's elements.
	Matrix& rows(int i, int j);

	Matrix& row(int rowIndex);

	Matrix& col(int colIndex);

	Matrix& transpose();

	void print();

	static Matrix& randomMatrix(int m, int n);

	static Matrix& randomMatrix(int m, int n, int bound);

};


inline Matrix& Matrix::mult(Matrix& A, Matrix& B) {
#ifdef USEGPU
	return gpuMult(A, B);
#else
	return cpuMult(A, B);
#endif
}


#endif
