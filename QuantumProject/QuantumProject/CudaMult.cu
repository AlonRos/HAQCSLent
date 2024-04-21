#include "CudaHeader.cuh"
#include <chrono>


typedef struct {
	int width;
	int height;
	int stride;
	double* elements;

} GpuMatrix;

__device__ double GetElement(GpuMatrix& A, int row, int col) {
	return A.elements[row * A.stride + col];
}
__device__ void SetElement(GpuMatrix& A, int row, int col, double value) {
	A.elements[row * A.stride + col] = value;
}

__device__ GpuMatrix GetSubMatrix(GpuMatrix& A, int row, int col, int blockHeightA, int blockWidthA)
{
	GpuMatrix Asub;
	Asub.width = blockWidthA;
	Asub.height = blockHeightA;
	Asub.stride = A.stride;
	Asub.elements = &A.elements[A.stride * blockHeightA * row + blockWidthA * col];
	return Asub;
}

__global__ void MatMulKernel(GpuMatrix A, GpuMatrix B, GpuMatrix C, int blockHeightA, int blockWidthAHeightB, int blockWidthB) {
	// Block row and column
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
	// Each thread block computes one sub-matrix Csub of C
	GpuMatrix Csub = GetSubMatrix(C, blockRow, blockCol, blockHeightA, blockWidthB);
	// Each thread computes one element of Csub
	// by accumulating results into Cvalue
	double Cvalue = 0;
	// Thread row and column within Csub
	int row = threadIdx.y;
	int col = threadIdx.x;

	// Loop over all the sub-matrices of A and B that are
	// required to compute Csub
	// Multiply each pair of sub-matrices together
	// and accumulate the results
	for (int i = 0; i < A.width / blockWidthAHeightB; ++i) {
		// Get sub-matrix Asub of A
		GpuMatrix Asub = GetSubMatrix(A, blockRow, i, blockHeightA, blockWidthAHeightB);
		// Get sub-matrix Bsub of B
		GpuMatrix Bsub = GetSubMatrix(B, i, blockCol, blockWidthAHeightB, blockWidthB);
		// Shared memory used to store Asub and Bsub respectively
		__shared__ double As[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE];
		__shared__ double Bs[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE];
		// Load Asub and Bsub from device memory to shared memory
		// Each thread loads one element of each sub-matrix

		if (row < Asub.height && col < Asub.width) {
			As[row][col] = GetElement(Asub, row, col);
		}

		if (row < Bsub.height && col < Bsub.width) {
			Bs[row][col] = GetElement(Bsub, row, col);
		}

		// Synchronize to make sure the sub-matrices are loaded
		// before starting the computation
		__syncthreads();
		// Multiply Asub and Bsub together
		
		if (row < Csub.height && col < Csub.width) {
			for (int j = 0; j < blockWidthAHeightB; ++j)
				Cvalue += As[row][j] * Bs[j][col];
		}

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}
	// Write Csub to device memory
	// Each thread writes one element
	if (row < Csub.height && col < Csub.width) {
		SetElement(Csub, row, col, Cvalue);
	}
}

__host__
double* gpuMultDouble(double* A, int Am, int An, double* B, int Bn) {
    GpuMatrix dev_A, dev_B, dev_res;
    int Bm = An, resm = Am, resn = Bn;

    dev_A.width = dev_A.stride = An;
    dev_A.height = Am;
    size_t size = dev_A.width * dev_A.height * sizeof(double);
    cudaMalloc(&dev_A.elements, size);
    cudaMemcpy(dev_A.elements, A, size, cudaMemcpyHostToDevice);

    dev_B.width = dev_B.stride = Bn;
    dev_B.height = Bm;
    size = dev_B.width * dev_B.height * sizeof(double);
    cudaMalloc(&dev_B.elements, size);
    cudaMemcpy(dev_B.elements, B, size, cudaMemcpyHostToDevice);


    dev_res.width = dev_res.stride = resn;
    dev_res.height = resm;
    size = dev_res.width * dev_res.height * sizeof(double);
    cudaMalloc(&dev_res.elements, size);


	int blockHeightA = min(MAX_BLOCK_SIZE, Am);
	int blockWidthAHeightB = min(MAX_BLOCK_SIZE, An);
	int blockWidthB = min(MAX_BLOCK_SIZE, Bn);

	dim3 dimBlock(max(blockWidthAHeightB, blockWidthB), max(blockHeightA, blockWidthAHeightB));

    dim3 dimGrid(ceil((double)dev_B.width / dimBlock.x), ceil((double)dev_A.height / dimBlock.y));

    MatMulKernel << <dimGrid, dimBlock >> > (dev_A, dev_B, dev_res, blockHeightA, blockWidthAHeightB, blockWidthB);

    double* res = (double*)malloc(size);

    cudaMemcpy(res, dev_res.elements, size, cudaMemcpyDeviceToHost);

    cudaFree(dev_A.elements);
    cudaFree(dev_B.elements);
    cudaFree(dev_res.elements);

    return res;
}

__host__
void Matrix::gpuMultIn(Matrix& A, Matrix& B, Matrix& res) {
	size_t size = A.m * A.n * sizeof(double);

	double* realA = (double*) malloc(size);
    double* imagA = (double*) malloc(size);


	for (int i = 0; i < A.m; ++i) {
		for (int j = 0; j < A.n; ++j) {
            realA[j + i * A.n] = A.entry(i, j).real();
            imagA[j + i * A.n] = A.entry(i, j).imag();
		}
	}

    size = B.m * B.n * sizeof(double);
    double* realB = (double*)malloc(size);
    double* imagB = (double*)malloc(size);

	
    for (int i = 0; i < B.m; ++i) {
        for (int j = 0; j < B.n; ++j) {
            realB[j + i * B.n] = B.entry(i, j).real();
            imagB[j + i * B.n] = B.entry(i, j).imag();
        }
    }

    double* realArealB = gpuMultDouble(realA, A.m, A.n, realB, B.n);
    double* imagAimagB = gpuMultDouble(imagA, A.m, A.n, imagB, B.n);

    double* realAimagB = gpuMultDouble(realA, A.m, A.n, imagB, B.n);
    double* imagArealB = gpuMultDouble(imagA, A.m, A.n, realB, B.n);
    
	for (int i = 0; i < A.m; ++i) {
		for (int j = 0; j < B.n; ++j) {
            res.entry(i, j) = complex_t(realArealB[j + i * B.n] - imagAimagB[j + i * B.n], realAimagB[j + i * B.n] + imagArealB[j + i * B.n]);
		}
	}
}