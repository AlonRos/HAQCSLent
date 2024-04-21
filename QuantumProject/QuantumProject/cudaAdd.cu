#include "CudaHeader.cuh"


__global__
void matAddKernel(cuDoubleComplex* A, cuDoubleComplex* B, cuDoubleComplex* res, int m, int n, int workPerThread, int threadUntil) {
	int threadIndex = threadIdx.x + threadIdx.y * blockDim.x;

	int i = min(threadIndex, threadUntil) * (workPerThread + 1) + max(threadIndex - threadUntil, 0) * workPerThread;

	int k;

	for (k = i; k < i + workPerThread; ++k) {
		res[k] = cuCadd(A[k], B[k]);
	}


	if (threadIndex < threadUntil) {
		res[k] = cuCadd(A[k], B[k]);
	}

}

__host__
void Matrix2::gpuAddIn(Matrix2& A, Matrix2& B, Matrix2& res) {
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
			A_vals[j + i * n] = complexToCudaComplex(A.entry(i, j));
		}
	}
	
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			B_vals[j + i * n] = complexToCudaComplex(B.entry(i, j));
		}
	}

	cuDoubleComplex* dev_A, * dev_B, * dev_res;

	cudaMalloc(&dev_A, A_valsLength);
	cudaMalloc(&dev_B, B_valsLength);
	cudaMalloc(&dev_res, res_valsLength);

	cudaMemcpy(dev_A, A_vals, A_valsLength, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B_vals, B_valsLength, cudaMemcpyHostToDevice);


	int number_of_threads = 1024;

	int number_of_blocks = 128;

	int workPerThread = (m * n) / number_of_threads;

	matAddKernel << < number_of_blocks, number_of_threads / number_of_blocks >> > (dev_A, dev_B, dev_res, m, n, workPerThread, (m * n) % number_of_threads);

	cudaMemcpy(res_vals, dev_res, res_valsLength, cudaMemcpyDeviceToHost);


	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			res.entry(i, j) = cudaComplexToComplex(res_vals[j + i * n]);
		}
	}

	free(A_vals);
	free(B_vals);
	free(res_vals);


	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_res);

}