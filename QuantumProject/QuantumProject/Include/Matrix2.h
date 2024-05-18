#ifndef Matrix2_H
#define Matrix2_H

#include <complex>
#include <format>
#include "../Include/Utils.h"

typedef std::complex<double> complex_t;

//#define USEGPU

class Matrix2 { // m * n Matrix
private:
	bool rowwise, toFree = true;
	int jump;

	static inline Matrix2& gpuMult(Matrix2& A, Matrix2& B);
	static inline Matrix2& cpuMult(Matrix2& A, Matrix2& B);
	static void gpuMultIn(Matrix2& A, Matrix2& B, Matrix2& saveIn);
	static void cpuMultIn(Matrix2& A, Matrix2& B, Matrix2& saveIn);

	static Matrix2& gpuAdd(Matrix2& A, Matrix2& B);
	static Matrix2& cpuAdd(Matrix2& A, Matrix2& B);
	static void cpuAddIn(Matrix2& A, Matrix2& B, Matrix2& saveIn);
	static void gpuAddIn(Matrix2& A, Matrix2& B, Matrix2& saveIn);

public:
	int m, n;
	complex_t* elements;

	// zeros elements
	Matrix2(int m, int n);

	// doesn't zero elements
	Matrix2(int m, int n, bool rowwise);

	// doesn't copy elements themselves
	Matrix2(int m, int n, complex_t* elements, bool rowwise, int jump);

	// copies elements themselves, assuming arr is a m * n array (rowwise)
	Matrix2(int m, int n, complex_t* arr);

	~Matrix2();

	// multiply matrices and return the result
	inline static Matrix2& mult(Matrix2& A, Matrix2& B);

	// multiply matrices and save the result in saveIn
	inline static void multIn(Matrix2& A, Matrix2& B, Matrix2& saveIn);

	// add matrices and return the result
	inline static Matrix2& add(Matrix2& A, Matrix2& B);

	// add matrices and save the result in saveIn
	inline static void addIn(Matrix2& A, Matrix2& B, Matrix2& saveIn);

	// kronecker-multiply matrices and return the result
	inline static Matrix2& kronecker(Matrix2& A, Matrix2& B);

	// kronecker-multiply matrices and save the result in saveIn
	static void kroneckerIn(Matrix2& A, Matrix2& B, Matrix2& saveIn);

	// multiply with other and return the result
	Matrix2& operator*(Matrix2& other);

	// add other and return the result
	Matrix2& operator+(Matrix2& other);

	// multiply by scalar and return the result
	Matrix2& operator*(complex_t scalar);

	// get the (rowIndex, colIndex) entry of the matrix
	complex_t& entry(int rowIndex, int colIndex);

	// zero the matrix elemenets
	void zero();

	// return a matrix containing the rows i to j (including i, not including j). The elements are in the same address as this' elements.
	Matrix2& rows(int i, int j);

	// return the rowIndex row
	inline Matrix2& row(int rowIndex);

	// return a matrix containing the columns i to j (including i, not including j). The elements are in the same address as this' elements.
	Matrix2& cols(int i, int j);

	// return the colIndex column
	inline Matrix2& col(int colIndex);

	// return the transpose of the matrix. The elements are in the same address as this' elements.
	Matrix2& transpose();

	// return the conjugate transpose of the matrix
	Matrix2& conjTranspose();

	// get the norm of the matrix squared
	double normSquared();

	// print the matrix
	void print();

	// get a random matrix with dimension m x n with elements a + ib, where 0 <= a, b < bound
	static Matrix2& randomMatrix(int m, int n, int bound);


};

inline Matrix2& Matrix2::row(int rowIndex) {
	return rows(rowIndex, rowIndex + 1);
}

inline Matrix2& Matrix2::col(int colIndex) {
	return cols(colIndex, colIndex + 1);
}

inline Matrix2& Matrix2::operator*(Matrix2& other) {
	return Matrix2::mult(*this, other);
}

inline Matrix2& Matrix2::operator+(Matrix2& other) {
	return Matrix2::add(*this, other);
}

Matrix2& createMatrixFromFunction(int* f, int length);

// multiply

inline Matrix2& Matrix2::mult(Matrix2& A, Matrix2& B) {
#ifdef USEGPU
	if ((long long)A.m * A.n * B.n > 729000 && B.n >= 4) {
		return gpuMult(A, B);
	}
	return cpuMult(A, B);
#else
	return cpuMult(A, B);
#endif
}


inline void Matrix2::multIn(Matrix2& A, Matrix2& B, Matrix2& saveIn) {
#ifdef USEGPU
	if ((long long)A.m * A.n * B.n > 729000 && B.n >= 4) {
		return gpuMultIn(A, B, saveIn);
	}
	return cpuMultIn(A, B, saveIn);
#else
	return cpuMultIn(A, B, saveIn);
#endif
}


inline Matrix2& Matrix2::cpuMult(Matrix2& A, Matrix2& B) {
	Matrix2* returnMatrix = new Matrix2(A.m, B.n);

	cpuMultIn(A, B, *returnMatrix);

	return *returnMatrix;
}

inline Matrix2& Matrix2::gpuMult(Matrix2& A, Matrix2& B) {
	Matrix2* returnMatrix = new Matrix2(A.m, B.n);

	gpuMultIn(A, B, *returnMatrix);

	return *returnMatrix;
}



// add

inline Matrix2& Matrix2::add(Matrix2& A, Matrix2& B) {
#ifdef USEGPU
	return gpuAdd(A, B);
#else
	return cpuAdd(A, B);
#endif
}

inline void Matrix2::addIn(Matrix2& A, Matrix2& B, Matrix2& saveIn) {
#ifdef USEGPU
	return gpuAddIn(A, B, saveIn);
#else
	return cpuAddIn(A, B, saveIn);
#endif
}

inline Matrix2& Matrix2::cpuAdd(Matrix2& A, Matrix2& B) {
	Matrix2* returnMatrix = new Matrix2(A.m, B.n);

	cpuAddIn(A, B, *returnMatrix);

	return *returnMatrix;
}

inline Matrix2& Matrix2::gpuAdd(Matrix2& A, Matrix2& B) {
	Matrix2* returnMatrix = new Matrix2(A.m, B.n);

	gpuAddIn(A, B, *returnMatrix);

	return *returnMatrix;
}


// kronecker

inline Matrix2& Matrix2::kronecker(Matrix2& A, Matrix2& B) {
	Matrix2* returnMatrix = new Matrix2(A.m * B.m, A.n * B.n);

	kroneckerIn(A, B, *returnMatrix);

	return *returnMatrix;
}



#endif