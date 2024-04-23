#include "CudaHeader.cuh"
#include <chrono>


typedef struct {
	int width;
	int height;
	int stride;
	cuDoubleComplex* elements;

} GpuMatrix;

__device__ cuDoubleComplex GetElement(GpuMatrix& A, int row, int col) {
	return A.elements[row * A.stride + col];
}
__device__ void SetElement(GpuMatrix& A, int row, int col, cuDoubleComplex value) {
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
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	GpuMatrix subC = GetSubMatrix(C, blockRow, blockCol, blockHeightA, blockWidthB);

	cuDoubleComplex Cvalue = make_cuDoubleComplex(0, 0);

	int row = threadIdx.y;
	int col = threadIdx.x;

	for (int i = 0; i < A.width / blockWidthAHeightB; ++i) {
		GpuMatrix subA = GetSubMatrix(A, blockRow, i, blockHeightA, blockWidthAHeightB);

		GpuMatrix subB = GetSubMatrix(B, i, blockCol, blockWidthAHeightB, blockWidthB);

		__shared__ cuDoubleComplex As[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE];
		__shared__ cuDoubleComplex Bs[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE];

		if (row < subA.height && col < subA.width) {
			As[row][col] = GetElement(subA, row, col);
		}

		if (row < subB.height && col < subB.width) {
			Bs[row][col] = GetElement(subB, row, col);
		}

		__syncthreads();
		
		if (row < subC.height && col < subC.width) {
			for (int j = 0; j < blockWidthAHeightB; ++j)
				Cvalue = cuCadd(Cvalue, cuCmul(As[row][j], Bs[j][col]));
		}

		__syncthreads();
	}

	if (row < subC.height && col < subC.width) {
		SetElement(subC, row, col, Cvalue);
	}
}

__host__
cuDoubleComplex* gpuMultComplex(cuDoubleComplex* A, int Am, int An, cuDoubleComplex* B, int Bn) {
    GpuMatrix dev_A, dev_B, dev_res;
    int Bm = An, resm = Am, resn = Bn;

    dev_A.width = dev_A.stride = An;
    dev_A.height = Am;
    size_t size = dev_A.width * dev_A.height * sizeof(cuDoubleComplex);
    cudaMalloc(&dev_A.elements, size);
    cudaMemcpy(dev_A.elements, A, size, cudaMemcpyHostToDevice);

    dev_B.width = dev_B.stride = Bn;
    dev_B.height = Bm;
    size = dev_B.width * dev_B.height * sizeof(cuDoubleComplex);
    cudaMalloc(&dev_B.elements, size);
    cudaMemcpy(dev_B.elements, B, size, cudaMemcpyHostToDevice);


    dev_res.width = dev_res.stride = resn;
    dev_res.height = resm;
    size = dev_res.width * dev_res.height * sizeof(cuDoubleComplex);
    cudaMalloc(&dev_res.elements, size);


	int blockHeightA = min(MAX_BLOCK_SIZE, Am);
	int blockWidthAHeightB = min(MAX_BLOCK_SIZE, An);
	int blockWidthB = min(MAX_BLOCK_SIZE, Bn);

	dim3 dimBlock(max(blockWidthAHeightB, blockWidthB), max(blockHeightA, blockWidthAHeightB));

    dim3 dimGrid(ceil((double)dev_B.width / dimBlock.x), ceil((double)dev_A.height / dimBlock.y));

    MatMulKernel << <dimGrid, dimBlock >> > (dev_A, dev_B, dev_res, blockHeightA, blockWidthAHeightB, blockWidthB);

    cuDoubleComplex* res = (cuDoubleComplex*)malloc(size);

    cudaMemcpy(res, dev_res.elements, size, cudaMemcpyDeviceToHost);

    cudaFree(dev_A.elements);
    cudaFree(dev_B.elements);
    cudaFree(dev_res.elements);

    return res;
}

__host__
void Matrix2::gpuMultIn(Matrix2& A, Matrix2& B, Matrix2& res) {
	size_t size = A.m * A.n * sizeof(cuDoubleComplex);

	cuDoubleComplex* Alement = (cuDoubleComplex*) malloc(size);


	for (int i = 0; i < A.m; ++i) {
		for (int j = 0; j < A.n; ++j) {
			Alement[j + i * A.n] = complexToCudaComplex(A.entry(i, j));
            //imagA[j + i * A.n] = A.entry(i, j).imag();
		}
	}

    size = B.m * B.n * sizeof(cuDoubleComplex);
    cuDoubleComplex* Blements = (cuDoubleComplex*)malloc(size);

	
    for (int i = 0; i < B.m; ++i) {
        for (int j = 0; j < B.n; ++j) {
			Blements[j + i * B.n] = complexToCudaComplex(B.entry(i, j));
        }
    }

    cuDoubleComplex* reslement = gpuMultComplex(Alement, A.m, A.n, Blements, B.n);
    
	for (int i = 0; i < A.m; ++i) {
		for (int j = 0; j < B.n; ++j) {
            res.entry(i, j) = cudaComplexToComplex(reslement[j + i * B.n]);
		}
	}
}

__host__
void init() {
	int* x;
	cudaMalloc(&x, sizeof(int));
	cudaFree(x);
}