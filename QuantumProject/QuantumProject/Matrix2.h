#ifndef Matrix2_H
#define Matrix2_H

#include <complex>
#include <format>
#include "Utils.h"

typedef std::complex<double> complex_t;

#define USEGPU

class Matrix2 { // m * n Matrix
private:
	bool rowwise;
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

	// Zeros elements
	Matrix2(int m, int n);

	// Doesn't zero elements
	Matrix2(int m, int n, bool rowwise);

	// Doesn't copy elements themselves
	Matrix2(int m, int n, complex_t* elements, bool rowwise, int jump);

	// Copies elements themselves, assuming arr is a m * n array (rowwise)
	Matrix2(int m, int n, complex_t* arr);

	~Matrix2();

	inline static Matrix2& mult(Matrix2& A, Matrix2& B);

	inline static void multIn(Matrix2& A, Matrix2& B, Matrix2& saveIn);

	inline static Matrix2& add(Matrix2& A, Matrix2& B);

	inline static void addIn(Matrix2& A, Matrix2& B, Matrix2& saveIn);

	Matrix2& operator*(Matrix2& other);

	Matrix2& operator+(Matrix2& other);

	Matrix2& operator*(complex_t scalar);

	complex_t& entry(int rowIndex, int colIndex);

	// Return a matrix containing the rows i to j (including i, not including j). The elements are in the same address as this's elements.
	Matrix2& rows(int i, int j);

	inline Matrix2& row(int rowIndex);

	Matrix2& cols(int i, int j);

	inline Matrix2& col(int colIndex);

	Matrix2& transpose();

	Matrix2& conjTranspose();

	double normSquared();

	void print();

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


// multiply

inline Matrix2& Matrix2::mult(Matrix2& A, Matrix2& B) {
#ifdef USEGPU
	if ((long long) A.m * A.n * B.m * B.n > 421875000) {
		return gpuMult(A, B);
	}
	return cpuMult(A, B);
#else
	return cpuMult(A, B);
#endif
}


inline void Matrix2::multIn(Matrix2& A, Matrix2& B, Matrix2& saveIn) {
#ifdef USEGPU
	if ((long long) A.m * A.n * B.m * B.n > 421875000) {
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

#endif