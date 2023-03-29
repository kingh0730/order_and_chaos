#include <iostream>
#include <cassert>

#include "position4x4_cuda.h"
#include "position4x4_masks.h"

__global__ void have_4_in_a_row(int *a, bool *b, int N)
{
    // Calculate global thread thread ID
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

    // Boundary check
    if (tid < N)
    {
        b[tid] = int_has_4_in_a_row(a[tid]);
    }
}

void test_have_4_in_a_row()
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

    // Threads per CTA (1024 threads per CTA)
    int BLOCK_SIZE = 1 << 10;

    // CTAs per Grid
    int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Call CUDA kernel
    have_4_in_a_row<<<GRID_SIZE, BLOCK_SIZE>>>(a, b, N);

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

    std::cout << "COMPLETED SUCCESSFULLY!\n";
}
