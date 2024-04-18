#include "CudaHeader.cuh"
#include <chrono>


typedef struct {
	int width;
	int height;
	int stride;
	double* elements;

} GpuMatrix;

// Get a matrix element
__device__ double GetElement(GpuMatrix& A, int row, int col)
{
	return A.elements[row * A.stride + col];
}
// Set a matrix element
__device__ void SetElement(GpuMatrix& A, int row, int col, double value)
{
	A.elements[row * A.stride + col] = value;
}
// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ GpuMatrix GetSubMatrix(GpuMatrix& A, int row, int col)
{
	GpuMatrix Asub;
	Asub.width = BLOCK_SIZE;
	Asub.height = BLOCK_SIZE;
	Asub.stride = A.stride;
	Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
	return Asub;
}


__global__ void MatMulKernel(GpuMatrix A, GpuMatrix B, GpuMatrix C)
{
	// Block row and column
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
	// Each thread block computes one sub-matrix Csub of C
	GpuMatrix Csub = GetSubMatrix(C, blockRow, blockCol);
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
	for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
		// Get sub-matrix Asub of A
		GpuMatrix Asub = GetSubMatrix(A, blockRow, m);
		// Get sub-matrix Bsub of B
		GpuMatrix Bsub = GetSubMatrix(B, m, blockCol);
		// Shared memory used to store Asub and Bsub respectively
		__shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];
		// Load Asub and Bsub from device memory to shared memory
		// Each thread loads one element of each sub-matrix
		As[row][col] = GetElement(Asub, row, col);
		Bs[row][col] = GetElement(Bsub, row, col);
		// Synchronize to make sure the sub-matrices are loaded
		// before starting the computation
		__syncthreads();
		// Multiply Asub and Bsub together
		for (int e = 0; e < BLOCK_SIZE; ++e)
			Cvalue += As[row][e] * Bs[e][col];

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}
	// Write Csub to device memory
	// Each thread writes one element
	SetElement(Csub, row, col, Cvalue);
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


    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(dev_B.width / dimBlock.x, dev_A.height / dimBlock.y);

    MatMulKernel << <dimGrid, dimBlock >> > (dev_A, dev_B, dev_res);

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