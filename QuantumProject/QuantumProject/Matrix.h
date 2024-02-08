#ifndef Matrix_H
#define Matrix_H

#include <complex>
#include "JumpArray.h"

typedef std::complex<double> complex_t;


class Matrix { // m * n Matrix
public:
	int m, n;
	JumpArray<complex_t>* elements; // Array of JumpArrays 
	bool jumpArrayIsRow;

	static Matrix gpuMult(Matrix& A, Matrix& B);
	static Matrix cpuMult(Matrix& A, Matrix& B);


public:

	// Zeros Every Entry
	Matrix(int m, int n, bool JAisRow);

	Matrix(int m, int n, bool JAisRow, JumpArray<complex_t> elements[]);

	inline static Matrix mult(Matrix& A, Matrix& B);

	Matrix operator*(Matrix& other);

	complex_t& entry(int rowIndex, int colIndex);

	Matrix row(int rowIndex);

	Matrix col(int colIndex);

};


inline Matrix Matrix::mult(Matrix& A, Matrix& B) {
#ifdef USEGPU
	return gpuMult(A, B);
#else
	return cpuMult(A, B);
#endif
}


#endif
