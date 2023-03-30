#include <iostream>
#include <cassert>

#include "position4x4_cuda.h"
#include "position4x4_masks.h"

__global__ void cuda_have_4_in_a_row(int *a, bool *b, int N)
{
    // Calculate global thread thread ID
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    // Boundary check
    if (tid < N)
    {
        uint32_t chars = a[tid];

        uint32_t fir_row = (chars & FIR_ROW_MASK);
        uint32_t sec_row = (chars & SEC_ROW_MASK);
        uint32_t thr_row = (chars & THR_ROW_MASK);
        uint32_t fou_row = (chars & FOU_ROW_MASK);

        uint32_t fir_col = (chars & FIR_COL_MASK);
        uint32_t sec_col = (chars & SEC_COL_MASK);
        uint32_t thr_col = (chars & THR_COL_MASK);
        uint32_t fou_col = (chars & FOU_COL_MASK);

        uint32_t pos_dia = (chars & POS_DIA_MASK);
        uint32_t neg_dia = (chars & NEG_DIA_MASK);

        b[tid] = (fir_row == FIR_ROW_OOOO || fir_row == FIR_ROW_XXXX ||
                  sec_row == SEC_ROW_OOOO || sec_row == SEC_ROW_XXXX ||
                  thr_row == THR_ROW_OOOO || thr_row == THR_ROW_XXXX ||
                  fou_row == FOU_ROW_OOOO || fou_row == FOU_ROW_XXXX ||
                  fir_col == FIR_COL_OOOO || fir_col == FIR_COL_XXXX ||
                  sec_col == SEC_COL_OOOO || sec_col == SEC_COL_XXXX ||
                  thr_col == THR_COL_OOOO || thr_col == THR_COL_XXXX ||
                  fou_col == FOU_COL_OOOO || fou_col == FOU_COL_XXXX ||
                  pos_dia == POS_DIA_OOOO || pos_dia == POS_DIA_XXXX ||
                  neg_dia == NEG_DIA_OOOO || neg_dia == NEG_DIA_XXXX);
    }
}

void test_cuda_have_4_in_a_row()
{
    // Array size of 2^16 (65536 elements)
    const int N = 1 << 16;

    // Declare unified memory pointers
    int *a;
    bool *b;

    // Allocation memory for these pointers
    cudaMallocManaged(&a, N * sizeof(int));
    cudaMallocManaged(&b, N * sizeof(bool));

    // Initialize vectors
    for (int i = 0; i < N; i++)
    {
        a[i] = rand() % 100;
    }

    // Call CUDA kernel
    cuda_have_4_in_a_row<<<GRID_SIZE(N), BLOCK_SIZE>>>(a, b, N);

    // Wait for all previous operations before using values
    // We need this because we don't get the implicit synchronization of
    // cudaMemcpy like in the original example
    cudaDeviceSynchronize();

    // Verify the result on the CPU
    for (int i = 0; i < N; i++)
    {
        assert(b[i] == int_has_4_in_a_row(a[i]));
    }

    // Free unified memory (same as memory allocated with cudaMalloc)
    cudaFree(a);
    cudaFree(b);

    std::cout << "CUDA test completed successfully!\n";
}

std::map<Position4x4, GameResult> cuda_solve_0_spaces_remain();
