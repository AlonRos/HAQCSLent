#ifndef Matrix_H
#define Matrix_H

#include <complex>
#include "JumpArray.h"

typedef std::complex<double> complex_t;
#define USEGPU


class Matrix { // m * n Matrix
private:
	JumpArray<complex_t>* elements; // Array of JumpArrays
	bool jumpArrayIsRow;

	static inline Matrix& gpuMult(Matrix& A, Matrix& B);
	static inline Matrix& cpuMult(Matrix& A, Matrix& B);
	static void gpuMultIn(Matrix& A, Matrix& B, Matrix& saveIn);
	static void cpuMultIn(Matrix& A, Matrix& B, Matrix& saveIn);

	static Matrix& gpuAdd(Matrix& A, Matrix& B);
	static Matrix& cpuAdd(Matrix& A, Matrix& B);
	static void cpuAddIn(Matrix& A, Matrix& B, Matrix& saveIn);
	static void gpuAddIn(Matrix& A, Matrix& B, Matrix& saveIn);

public:
	int m, n;

	// Zeros Every Entry
	Matrix(int m, int n, bool JAisRow);

	Matrix(int m, int n);

	Matrix(int m, int n, bool JAisRow, JumpArray<complex_t> elements[]);

	static Matrix& fromArray(int m, int n, bool JAisRow, complex_t* arr);

	static Matrix& fromArray(int m, int n, complex_t* arr);

	inline static Matrix& mult(Matrix& A, Matrix& B);

	inline static void multIn(Matrix& A, Matrix& B, Matrix& saveIn);

	inline static Matrix& add(Matrix& A, Matrix& B);

	inline static void addIn(Matrix& A, Matrix& B, Matrix& saveIn);

	Matrix& operator*(Matrix& other);

	Matrix& operator+(Matrix& other);

	Matrix& operator*(complex_t scalar);

	complex_t& entry(int rowIndex, int colIndex);

	// Return a matrix containing the rows i to j (including i, not including j). The elements are in the same address as this's elements.
	Matrix& rows(int i, int j);

	Matrix& row(int rowIndex);

	Matrix& cols(int i, int j);

	Matrix& col(int colIndex);

	Matrix& transpose();

	void print();

	static Matrix& randomMatrix(int m, int n);

	static Matrix& randomMatrix(int m, int n, int bound);

};

inline Matrix& Matrix::operator*(Matrix& other) {
	return Matrix::mult(*this, other);
}

inline Matrix& Matrix::operator+(Matrix& other) {
	return Matrix::add(*this, other);
}


// multiply

inline Matrix& Matrix::mult(Matrix& A, Matrix& B) {
#ifdef USEGPU
	return gpuMult(A, B);
#else
	return cpuMult(A, B);
#endif
}


inline void Matrix::multIn(Matrix& A, Matrix& B, Matrix& saveIn) {
#ifdef USEGPU

	return gpuMultIn(A, B, saveIn);
#else
	return cpuMultIn(A, B, saveIn);
#endif
}


inline Matrix& Matrix::cpuMult(Matrix& A, Matrix& B) {
	Matrix* returnMatrix = new Matrix(A.m, B.n);

	cpuMultIn(A, B, *returnMatrix);

	return *returnMatrix;
}

inline Matrix& Matrix::gpuMult(Matrix& A, Matrix& B) {
	Matrix* returnMatrix = new Matrix(A.m, B.n);

	gpuMultIn(A, B, *returnMatrix);

	return *returnMatrix;
}



// add

inline Matrix& Matrix::add(Matrix& A, Matrix& B) {
#ifdef USEGPU
	return gpuAdd(A, B);
#else
	return cpuAdd(A, B);
#endif
}

inline void Matrix::addIn(Matrix& A, Matrix& B, Matrix& saveIn) {
#ifdef USEGPU
	return gpuAddIn(A, B, saveIn);
#else
	return cpuAddIn(A, B, saveIn);
#endif
}

inline Matrix& Matrix::cpuAdd(Matrix& A, Matrix& B) {
	Matrix* returnMatrix = new Matrix(A.m, B.n);

	cpuAddIn(A, B, *returnMatrix);

	return *returnMatrix;
}

inline Matrix& Matrix::gpuAdd(Matrix& A, Matrix& B) {
	Matrix* returnMatrix = new Matrix(A.m, B.n);

	gpuAddIn(A, B, *returnMatrix);

	return *returnMatrix;
}





#endif
