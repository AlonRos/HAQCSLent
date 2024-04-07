#include "CudaHeader.cuh"
#include <chrono>

#define BLOCK_SIZE 16

typedef struct {
	int width;
	int height;
	int stride;
	float* elements;

} GpuMatrix;

template <size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
    size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS, size_t BLOCK_TILE_SKEW_SIZE_X = 0U, size_t BLOCK_TILE_SKEW_SIZE_K = 0U>
__device__ void load_data_to_shared_memory(float const* A, size_t lda,
    float const* B, size_t ldb,
    float A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K + BLOCK_TILE_SKEW_SIZE_K],
    float B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
    size_t thread_block_tile_idx,
    size_t thread_linear_idx,
    size_t m, size_t n,
    size_t k)
{
    // Load data from A on DRAM to A_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx{ 0U };
        load_idx <
        (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) / NUM_THREADS; ++load_idx)
    {
        size_t const A_thread_block_tile_row_idx{(thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_K };
        size_t const A_thread_block_tile_col_idx{(thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_K };
        size_t const A_row_idx{ blockIdx.y * BLOCK_TILE_SIZE_Y + A_thread_block_tile_row_idx };
        size_t const A_col_idx{ thread_block_tile_idx * BLOCK_TILE_SIZE_K + A_thread_block_tile_col_idx };

        // These boundary checks might slow down the kernel to some extent.
        // But they guarantee the correctness of the kernel for all
        // different GEMM configurations.
        float val{ 0 };
        if (A_row_idx < m && A_col_idx < k)
        {
            val = A[A_row_idx * lda + A_col_idx];
        }
        // This if will slow down the kernel.
        // Add static asserts from the host code to guarantee this if is
        // always true.
        //static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS == 0U);
        // if (A_thread_block_tile_row_idx < BLOCK_TILE_SIZE_Y &&
        //     A_thread_block_tile_col_idx < BLOCK_TILE_SIZE_K)
        // {
        //     A_thread_block_tile[A_thread_block_tile_row_idx]
        //                        [A_thread_block_tile_col_idx] = val;
        // }
        A_thread_block_tile[A_thread_block_tile_row_idx][A_thread_block_tile_col_idx] = val;
    }
    // Load data from B on DRAM to B_thread_block_tile on shared memory.
#pragma unroll
    for (size_t load_idx{ 0U };
        load_idx <
        (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X + NUM_THREADS - 1U) /
        NUM_THREADS;
        ++load_idx)
    {
        size_t const B_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) /
            BLOCK_TILE_SIZE_X };
        size_t const B_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) %
            BLOCK_TILE_SIZE_X };
        size_t const B_row_idx{ thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                               B_thread_block_tile_row_idx };
        size_t const B_col_idx{ blockIdx.x * BLOCK_TILE_SIZE_X +
                               B_thread_block_tile_col_idx };

        // These boundary checks might slow down the kernel to some extent.
        // But they guarantee the correctness of the kernel for all
        // different GEMM configurations.
        float val{ 0 };
        if (B_row_idx < k && B_col_idx < n)
        {
            val = B[B_row_idx * ldb + B_col_idx];
        }
        // This if will slow down the kernel.
        // Add static asserts from the host code to guarantee this if is
        // always true.
        //static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS == 0U);
        // if (B_thread_block_tile_row_idx < BLOCK_TILE_SIZE_K &&
        //     B_thread_block_tile_col_idx < BLOCK_TILE_SIZE_X)
        // {
        //     B_thread_block_tile[B_thread_block_tile_row_idx]
        //                        [B_thread_block_tile_col_idx] = val;
        // }
        B_thread_block_tile[B_thread_block_tile_row_idx]
            [B_thread_block_tile_col_idx] = val;
    }
}


template <size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
    size_t BLOCK_TILE_SIZE_K, size_t THREAD_TILE_SIZE_X,
    size_t THREAD_TILE_SIZE_Y>
__global__ void gemm_v04(size_t m, size_t n, size_t k, GpuMatrix A, size_t lda, GpuMatrix B, size_t ldb, GpuMatrix C, size_t ldc) {
    // Avoid using blockDim.x * blockDim.y as the number of threads per block.
    // Because it is a runtime constant and the compiler cannot optimize the
    // loop unrolling based on that.
    // Use a compile time constant instead.
    constexpr size_t NUM_THREADS{ BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
                                 (THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y) };
    size_t const thread_linear_idx{ threadIdx.y * blockDim.x + threadIdx.x };

    // Cache a tile of A and B in shared memory for data reuse.
    __shared__ float A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ float B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    size_t const num_thread_block_tiles{ (k + BLOCK_TILE_SIZE_K - 1) /
                                        BLOCK_TILE_SIZE_K };

    // Each thread in the block processes BLOCK_TILE_SIZE_Y output values.
    // Specifically, these values corresponds to
    // C[blockIdx.y * BLOCK_TILE_SIZE_Y + threadIdx.x / BLOCK_TILE_SIZE_X *
    // THREAD_TILE_SIZE_Y : blockIdx.y * BLOCK_TILE_SIZE_Y + (threadIdx.x /
    // BLOCK_TILE_SIZE_X + 1) * THREAD_TILE_SIZE_Y][blockIdx.x *
    // BLOCK_TILE_SIZE_X + threadIdx.x % BLOCK_TILE_SIZE_X *
    // THREAD_TILE_SIZE_X : blockIdx.x * BLOCK_TILE_SIZE_X + (threadIdx.x %
    // BLOCK_TILE_SIZE_X + 1) * THREAD_TILE_SIZE_X]
    float C_thread_results[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = { 0 };
    // A_vals is cached in the register.
    float A_vals[THREAD_TILE_SIZE_Y] = { 0 };
    // B_vals is cached in the register.
    float B_vals[THREAD_TILE_SIZE_X] = { 0 };

    for (size_t thread_block_tile_idx{ 0U };
        thread_block_tile_idx < num_thread_block_tiles;
        ++thread_block_tile_idx)
    {

        load_data_to_shared_memory<BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
            BLOCK_TILE_SIZE_K, NUM_THREADS>(
                A.elements, lda, B.elements, ldb, A_thread_block_tile, B_thread_block_tile,
                thread_block_tile_idx, thread_linear_idx, m, n, k);
        __syncthreads();

#pragma unroll
        for (size_t k_i{ 0U }; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
            size_t const A_thread_block_tile_row_idx{
                thread_linear_idx / (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                THREAD_TILE_SIZE_Y };
            size_t const A_thread_block_tile_col_idx{ k_i };

#pragma unroll
            for (size_t thread_tile_row_idx{ 0U };
                thread_tile_row_idx < THREAD_TILE_SIZE_Y;
                ++thread_tile_row_idx)
            {
                // There will be shared memory bank conflicts accessing the
                // values from A_thread_block_tile. We can do it better by
                // transposing the A_thread_block_tile when we load the data
                // from DRAM.
                A_vals[thread_tile_row_idx] =
                    A_thread_block_tile[A_thread_block_tile_row_idx +
                    thread_tile_row_idx]
                    [A_thread_block_tile_col_idx];
            }

            size_t const B_thread_block_tile_row_idx{ k_i };
            size_t const B_thread_block_tile_col_idx{
                thread_linear_idx % (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) *
                THREAD_TILE_SIZE_X };
#pragma unroll
            for (size_t thread_tile_col_idx{ 0U };
                thread_tile_col_idx < THREAD_TILE_SIZE_X;
                ++thread_tile_col_idx)
            {
                B_vals[thread_tile_col_idx] =
                    B_thread_block_tile[B_thread_block_tile_row_idx]
                    [B_thread_block_tile_col_idx +
                    thread_tile_col_idx];
            }

            for (size_t thread_tile_row_idx{ 0U };
                thread_tile_row_idx < THREAD_TILE_SIZE_Y;
                ++thread_tile_row_idx)
            {
                for (size_t thread_tile_col_idx{ 0U };
                    thread_tile_col_idx < THREAD_TILE_SIZE_X;
                    ++thread_tile_col_idx)
                {
                    C_thread_results[thread_tile_row_idx][thread_tile_col_idx] = cuCadd(C_thread_results[thread_tile_row_idx][thread_tile_col_idx], cuCmul(A_vals[thread_tile_row_idx], B_vals[thread_tile_col_idx]));
                }
            }
        }
        __syncthreads();
    }

    // Write the results to DRAM.
    for (size_t thread_tile_row_idx{ 0U };
        thread_tile_row_idx < THREAD_TILE_SIZE_Y; ++thread_tile_row_idx)
    {
        for (size_t thread_tile_col_idx{ 0U };
            thread_tile_col_idx < THREAD_TILE_SIZE_X; ++thread_tile_col_idx)
        {
            size_t const C_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y + threadIdx.x / (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_Y + thread_tile_row_idx };
            size_t const C_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X + threadIdx.x % (BLOCK_TILE_SIZE_X / THREAD_TILE_SIZE_X) * THREAD_TILE_SIZE_X + thread_tile_col_idx };
            
            if (C_row_idx < m && C_col_idx < n)
            {
                C.elements[C_row_idx * ldc + C_col_idx] = cuCadd(C_thread_results[thread_tile_row_idx][thread_tile_col_idx], C.elements[C_row_idx * ldc + C_col_idx]);
            }
        }
    }
}





























































// Get a matrix element
__device__ float GetElement(GpuMatrix& A, int row, int col)
{
	return A.elements[row * A.stride + col];
}
// Set a matrix element
__device__ void SetElement(GpuMatrix& A, int row, int col, float value)
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

//__global__
//void matMultKernel(float* A, int Am, int An, float* B, int Bm, int Bn, float* res) {
//
//	float Cvalue = make_cuDoubleComplex(0, 0);
//	int row = blockIdx.y * blockDim.y + threadIdx.y;
//	int col = blockIdx.x * blockDim.x + threadIdx.x;
//	for (int e = 0; e < An; ++e)
//		Cvalue = cuCadd(Cvalue, cuCmul(getElement(A, row, e, An), getElement(B, e, col, Bn)));
//
//	res[row * Bn + col] = Cvalue;
//
//}

__global__ void MatMulKernel(GpuMatrix A, GpuMatrix B, GpuMatrix C)
{
	// Block row and column
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
	// Each thread block computes one sub-matrix Csub of C
	GpuMatrix Csub = GetSubMatrix(C, blockRow, blockCol);
	// Each thread computes one element of Csub
	// by accumulating results into Cvalue
	float Cvalue = 0;
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
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
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















































#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

typedef struct {
    float x, y, z, w;
} cuDoubleComplex4;


typedef unsigned int uint;

const int WARPSIZE = 32; // warpSize is not constexpr

template <const int BM, const int BN, const int BK, const int rowStrideA,
    const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float* A, const float* B,
    float* As, float* Bs, int innerRowA, int innerColA,
    int innerRowB, int innerColB) {
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
        const cuDoubleComplex4 tmp = reinterpret_cast<const cuDoubleComplex4*>(
            &A[(innerRowA + offset) * K + innerColA * 4])[0];
        // float4 tmp;
        // asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
        //     : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
        //     : "l"(&A[(innerRowA + offset) * K + innerColA * 4]));
        As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
        reinterpret_cast<cuDoubleComplex4*>(
            &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
            reinterpret_cast<const cuDoubleComplex4*>(
                &B[(innerRowB + offset) * N + innerColB * 4])[0];
        // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
        //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
        //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
        //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
        //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
        //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
    }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
    const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
    const int TM, const int TN>
__device__ void
    processFromSmem(float* regM, float* regN, float* threadResults, const float* As,
        const float* Bs, const uint warpRow, const uint warpCol,
        const uint threadRowInWarp, const uint threadColInWarp) {
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
        // populate registers for whole warptile
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (uint i = 0; i < TM; ++i) {
                regM[wSubRowIdx * TM + i] =
                    As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
                    threadRowInWarp * TM + i];
            }
        }
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            for (uint i = 0; i < TN; ++i) {
                regN[wSubColIdx * TN + i] =
                    Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
                    threadColInWarp * TN + i];
            }
        }

        // execute warptile matmul
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                // calculate per-thread results
                for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                        threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) + (wSubColIdx * TN) + resIdxN] += regM[wSubRowIdx * TM + resIdxM] * regN[wSubColIdx * TN + resIdxN];
                    }
                }
            }
        }
    }
}


/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN, const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) sgemmWarptiling(int M, int N, int K, float* A, float* B, float* C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // Placement of the warp in the threadblock tile
    const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
    const uint warpCol = warpIdx % (BN / WN);
    const uint warpRow = warpIdx / (BN / WN);

    // size of the warp subtile
    constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr uint WSUBM = WM / WMITER; // 64/2=32
    constexpr uint WSUBN = WN / WNITER; // 32/2=16

    // Placement of the thread in the warp subtile
    const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

    // allocate space for the current blocktile in SMEM
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    // Move C_ptr to warp's output tile
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

    // allocate thread-local cache for results in registerfile
    float threadResults[WMITER * TM * WNITER * TN] = { 0 };
    // we cache into registers on the warptile level
    float regM[WMITER * TM] = { 0 };
    float regN[WNITER * TN] = { 0 };

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
            N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
        __syncthreads();
        processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
            TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,
                threadRowInWarp, threadColInWarp);
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down
        __syncthreads();
    }

    // write out the results
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            // move C pointer to current warp subtile
            float* C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    // load C vector into registers
                    cuDoubleComplex4 tmp = reinterpret_cast<cuDoubleComplex4*>(
                        &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                        threadColInWarp * TN + resIdxN])[0];
                    // perform GEMM update in reg
                    const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
                    tmp.x = threadResults[i + 0] + tmp.x;
                    tmp.y = threadResults[i + 1] + tmp.y;
                    tmp.z = threadResults[i + 2] + tmp.z;
                    tmp.w = threadResults[i + 3] + tmp.w;
                    // write back
                    reinterpret_cast<cuDoubleComplex4*>(
                        &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                        threadColInWarp * TN + resIdxN])[0] = tmp;
                }
            }
        }
    }
}





























__host__
void Matrix::gpuMultIn(Matrix& A, Matrix& B, Matrix& res) {
	GpuMatrix dev_A, dev_B, dev_res;

	dev_A.width = dev_A.stride = A.n;
	dev_A.height = A.m;

	size_t size = dev_A.width * dev_A.height * sizeof(float);
	cudaMalloc(&dev_A.elements, size);

	float* tempA = (float*) malloc(size);

	for (int i = 0; i < A.m; ++i) {
		for (int j = 0; j < A.n; ++j) {
			tempA[j + i * A.n] = A.entry(i, j).real();
		}
	}

	cudaMemcpy(dev_A.elements, tempA, size, cudaMemcpyHostToDevice);


	dev_B.width = dev_B.stride = B.n;
	dev_B.height = B.m;

	size = dev_B.width * dev_B.height * sizeof(float);
	cudaMalloc(&dev_B.elements, size);

	float* tempB = (float*)malloc(size);

	for (int i = 0; i < B.m; ++i) {
		for (int j = 0; j < B.n; ++j) {
			tempB[j + i * B.n] = B.entry(i, j).real();
		}
	}

	cudaMemcpy(dev_B.elements, tempB, size, cudaMemcpyHostToDevice);



	dev_res.width = dev_res.stride = res.n;
	dev_res.height = res.m;

	size = dev_res.width * dev_res.height * sizeof(float);
	cudaMalloc(&dev_res.elements, size);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(dev_B.width / dimBlock.x, dev_A.height / dimBlock.y);

	MatMulKernel << <dimGrid, dimBlock >> > (dev_A, dev_B, dev_res);  // doing 1024*1024 in 1000ms 

    //constexpr unsigned int BLOCK_TILE_SIZE_X{ 64 };
    //constexpr unsigned int BLOCK_TILE_SIZE_Y{ 128 };
    //constexpr unsigned int BLOCK_TILE_SIZE_K{ 16 };
    //// Each thread computes THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y values of C.
    //constexpr unsigned int THREAD_TILE_SIZE_X{ 8U };
    //constexpr unsigned int THREAD_TILE_SIZE_Y{ 8U };
    //constexpr unsigned int NUM_THREADS_PER_BLOCK{
    //    BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y /
    //    (THREAD_TILE_SIZE_X * THREAD_TILE_SIZE_Y) };

    int m = A.m, k = A.n, n = B.n;

    //dim3 const block_dim{ NUM_THREADS_PER_BLOCK, 1U, 1U };
    //dim3 const grid_dim {
    //    (static_cast<unsigned int>(n) + BLOCK_TILE_SIZE_X - 1U) / BLOCK_TILE_SIZE_X,
    //    (static_cast<unsigned int>(m) + BLOCK_TILE_SIZE_Y - 1U) / BLOCK_TILE_SIZE_Y,
    //    1U };
    //
    //gemm_v04<BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
    //    THREAD_TILE_SIZE_X, THREAD_TILE_SIZE_Y > <<< grid_dim, block_dim >>> (
    //        m, n, k, dev_A, A.m, dev_B, B.m, dev_res, res.m);

    //int M = m, N = n, K = k;
    //
    //const uint K10_NUM_THREADS = 128;
    //const uint K10_BN = 128;
    //const uint K10_BM = 64;
    //const uint K10_BK = 16;
    //const uint K10_WN = 64;
    //const uint K10_WM = 64;
    //const uint K10_WNITER = 4;
    //const uint K10_TN = 4;
    //const uint K10_TM = 8;
    //
    //
    //dim3 blockDim(K10_NUM_THREADS);
    //dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(M, K10_BM));
    //sgemmWarptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
    //    K10_TN, K10_NUM_THREADS>
    //    << <gridDim, blockDim >> > (M, N, K, dev_A.elements, dev_B.elements, dev_res.elements);


    printf("Error: %d\n", cudaGetLastError());

	float* tempRes = (float*) malloc(size);

	cudaMemcpy(tempRes, dev_res.elements, size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < A.m; ++i) {
		for (int j = 0; j < B.n; ++j) {
			res.entry(i, j) = tempRes[j + i * B.n];
		}
	}

	// Free device memory
	cudaFree(dev_A.elements);
	cudaFree(dev_B.elements);
	cudaFree(dev_res.elements);
}