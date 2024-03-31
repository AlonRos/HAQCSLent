#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include "CudaHeader.cuh"

__host__ 
inline cuDoubleComplex complexToCudaComplex(complex_t z) {
	return make_cuDoubleComplex(z.real(), z.imag());
}

__host__ 
inline complex_t cudaComplexToComplex(cuDoubleComplex z) {
	return complex_t(z.x, z.y);
}


__global__
void matAdd(cuDoubleComplex* A, cuDoubleComplex*B, cuDoubleComplex* res, int m, int n, int workPerThread, int threadUntil) {

	int i = min(threadIdx.x, threadUntil) * (workPerThread + 1) + max(threadIdx.x - threadUntil, 0) * workPerThread;

	int k;

	for (k = i; k < i + workPerThread; ++k) {
		res[i + k] = cuCadd(A[i + k], B[i + k]);
	}

	if (threadIdx.x < threadUntil) {
		res[i + k] = cuCadd(A[i + k], B[i + k]);
	}

}



__host__
void Matrix::gpuAddIn(Matrix& A, Matrix& B, Matrix& res) {
	cuDoubleComplex* A_vals, * B_vals, *res_vals;

	int m = A.m, n = A.n;

	int A_valsLength = m * n * sizeof(cuDoubleComplex);
	int B_valsLength = A_valsLength;
	int res_valsLength = A_valsLength;

	A_vals = (cuDoubleComplex*)malloc(A_valsLength);
	B_vals = (cuDoubleComplex*)malloc(B_valsLength);
	res_vals = (cuDoubleComplex*)malloc(res_valsLength);


	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			A_vals[i + m * j] = complexToCudaComplex(A.entry(i, j));
		}
	}

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			B_vals[i + m * j] = complexToCudaComplex(B.entry(i, j));
		}
	}

	cuDoubleComplex* dev_A, * dev_B, * dev_res;

	cudaMalloc(&dev_A, A_valsLength);
	cudaMalloc(&dev_B, B_valsLength);
	cudaMalloc(&dev_res, res_valsLength);

	cudaMemcpy(dev_A, A_vals, A_valsLength, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B_vals , B_valsLength, cudaMemcpyHostToDevice);


	int numbr_of_threads = 80;

	int workPerThread = m * n / numbr_of_threads;

	matAdd <<< n, numbr_of_threads >>> (dev_A, dev_B, dev_res, m, n, workPerThread, m * n % numbr_of_threads);

	cudaMemcpy(res_vals, dev_res, res_valsLength, cudaMemcpyDeviceToHost);


	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			res.entry(i, j) = cudaComplexToComplex(res_vals[i + m * j]);
		}
	}
}


__global__
void matMult(cuDoubleComplex* A, int Am, int An, cuDoubleComplex* B, int Bm, int Bn, cuDoubleComplex* res) {
	// i, k, j

}

__host__
void Matrix::gpuMultIn(Matrix& A, Matrix& B, Matrix& res) {
	cuDoubleComplex* A_vals, * B_vals, * res_vals;

	int m = A.m, n = A.n;

	int A_valsLength = m * n * sizeof(cuDoubleComplex);
	int B_valsLength = A_valsLength;
	int res_valsLength = A_valsLength;

	A_vals = (cuDoubleComplex*)malloc(A_valsLength);
	B_vals = (cuDoubleComplex*)malloc(B_valsLength);
	res_vals = (cuDoubleComplex*)malloc(res_valsLength);


	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			A_vals[i + m * j] = complexToCudaComplex(A.entry(i, j));
		}
	}

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			B_vals[i + m * j] = complexToCudaComplex(B.entry(i, j));
		}
	}

	cuDoubleComplex* dev_A, * dev_B, * dev_res;

	cudaMalloc(&dev_A, A_valsLength);
	cudaMalloc(&dev_B, B_valsLength);
	cudaMalloc(&dev_res, res_valsLength);

	cudaMemcpy(dev_A, A_vals, A_valsLength, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B_vals, B_valsLength, cudaMemcpyHostToDevice);


	int numbr_of_threads = 80;

	int workPerThread = m * n / numbr_of_threads;

	matMult << < n, numbr_of_threads >> > (dev_A, dev_B, dev_res, m, n, workPerThread, m * n % numbr_of_threads);

	cudaMemcpy(res_vals, dev_res, res_valsLength, cudaMemcpyDeviceToHost);


	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			res.entry(i, j) = cudaComplexToComplex(res_vals[i + m * j]);
		}
	}
}