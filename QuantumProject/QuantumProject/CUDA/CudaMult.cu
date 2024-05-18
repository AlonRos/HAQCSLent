#include "../CUDA/CudaHeader.cuh"
#include <chrono>


typedef struct {
	int width;
	int height;
	int jump;
	cuDoubleComplex* elements;

} GpuMatrix;

__device__ 
cuDoubleComplex getElement(GpuMatrix& A, int row, int col) {
	return A.elements[row * A.jump + col];
}
__device__ 
void setElement(GpuMatrix& A, int row, int col, cuDoubleComplex value) {
	A.elements[row * A.jump + col] = value;
}

__device__ 
GpuMatrix getSubMatrix(GpuMatrix& A, int row, int col, int blockHeightA, int blockWidthA)
{
	GpuMatrix Asub;
	Asub.width = blockWidthA;
	Asub.height = blockHeightA;
	Asub.jump = A.jump;
	Asub.elements = &A.elements[A.jump * blockHeightA * row + blockWidthA * col]; // staritng in the element (row, col)
	return Asub;
}

__global__ 
void matMulKernel(GpuMatrix A, GpuMatrix B, GpuMatrix C, int blockHeightA, int blockWidthAHeightB, int blockWidthB) {
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	GpuMatrix subC = getSubMatrix(C, blockRow, blockCol, blockHeightA, blockWidthB);

	cuDoubleComplex Cvalue = make_cuDoubleComplex(0, 0);

	int row = threadIdx.y;
	int col = threadIdx.x;

	__shared__ cuDoubleComplex As[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE]; // shared sub matrix of A
	__shared__ cuDoubleComplex Bs[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE]; // shared sub matrix of B

	for (int i = 0; i < A.width / blockWidthAHeightB; ++i) {
		GpuMatrix subA = getSubMatrix(A, blockRow, i, blockHeightA, blockWidthAHeightB); // get the current sub matrix of A

		GpuMatrix subB = getSubMatrix(B, i, blockCol, blockWidthAHeightB, blockWidthB); // get the current sub matrix of B

		if (row < subA.height && col < subA.width) {
			As[row][col] = getElement(subA, row, col); // copy from subA to the shared memory
		}

		if (row < subB.height && col < subB.width) {
			Bs[row][col] = getElement(subB, row, col); // copy from subB to the shared memory
		}

		__syncthreads(); // wait for all threads

		if (row < subC.height && col < subC.width) {
			for (int j = 0; j < blockWidthAHeightB; ++j) { // compute the row times the col
				Cvalue = cuCadd(Cvalue, cuCmul(As[row][j], Bs[j][col]));
			}
		}

		__syncthreads(); // wait for all threads
	}

	if (row < subC.height && col < subC.width) {
		setElement(subC, row, col, Cvalue); // save the result
	}
}

__host__
cuDoubleComplex* gpuMultArrs(cuDoubleComplex* A, int Am, int An, cuDoubleComplex* B, int Bn) {
	GpuMatrix dev_A, dev_B, dev_res;
	int Bm = An, resm = Am, resn = Bn;

	// copy cuDoubleComplex* A to GpuMatrix dev_A
	dev_A.width = dev_A.jump = An;
	dev_A.height = Am;
	size_t size = dev_A.width * dev_A.height * sizeof(cuDoubleComplex);
	cudaMalloc(&dev_A.elements, size);
	cudaMemcpy(dev_A.elements, A, size, cudaMemcpyHostToDevice);

	// copy cuDoubleComplex* B to GpuMatrix dev_B
	dev_B.width = dev_B.jump = Bn;
	dev_B.height = Bm;
	size = dev_B.width * dev_B.height * sizeof(cuDoubleComplex);
	cudaMalloc(&dev_B.elements, size);
	cudaMemcpy(dev_B.elements, B, size, cudaMemcpyHostToDevice);


	// alloc gpu memory for the result
	dev_res.width = dev_res.jump = resn;
	dev_res.height = resm;
	size = dev_res.width * dev_res.height * sizeof(cuDoubleComplex);
	cudaMalloc(&dev_res.elements, size);


	// calculate the block size
	int blockHeightA = min(MAX_BLOCK_SIZE, Am);
	int blockWidthAHeightB = min(MAX_BLOCK_SIZE, An);
	int blockWidthB = min(MAX_BLOCK_SIZE, Bn);

	dim3 dimBlock(max(blockWidthAHeightB, blockWidthB), max(blockHeightA, blockWidthAHeightB));

	// calculate the grid size
	dim3 dimGrid((int)ceil((double)dev_B.width / dimBlock.x), (int)ceil((double)dev_A.height / dimBlock.y));

	matMulKernel <<< dimGrid, dimBlock >>> (dev_A, dev_B, dev_res, blockHeightA, blockWidthAHeightB, blockWidthB); // calling the kernel

	cuDoubleComplex* res = (cuDoubleComplex*)malloc(size);

	// copy the result to the given array
	cudaMemcpy(res, dev_res.elements, size, cudaMemcpyDeviceToHost);

	cudaFree(dev_A.elements);
	cudaFree(dev_B.elements);
	cudaFree(dev_res.elements);

	return res;
}

__host__
void Matrix2::gpuMultIn(Matrix2& A, Matrix2& B, Matrix2& res) {

	// copy the elements from A to an array
	size_t size = A.m * A.n * sizeof(cuDoubleComplex);
	cuDoubleComplex* Alements = (cuDoubleComplex*)malloc(size);

	for (int i = 0; i < A.m; ++i) {
		for (int j = 0; j < A.n; ++j) {
			Alements[j + i * A.n] = complexToCudaComplex(A.entry(i, j));
		}
	}

	// copy the elements from B to an array
	size = B.m * B.n * sizeof(cuDoubleComplex);
	cuDoubleComplex* Blements = (cuDoubleComplex*)malloc(size);


	for (int i = 0; i < B.m; ++i) {
		for (int j = 0; j < B.n; ++j) {
			Blements[j + i * B.n] = complexToCudaComplex(B.entry(i, j));
		}
	}

	cuDoubleComplex* reslements = gpuMultArrs(Alements, A.m, A.n, Blements, B.n);

	free(Alements);
	free(Blements);


	// copy the result from the array to res
	for (int i = 0; i < A.m; ++i) {
		for (int j = 0; j < B.n; ++j) {
			res.entry(i, j) = cudaComplexToComplex(reslements[j + i * B.n]);
		}
	}

	free(reslements);
}

__host__
void init() {
	int* x;
	cudaMalloc(&x, sizeof(int));
	cudaFree(x);
}