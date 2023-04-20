#include <chrono>
#include <iostream>
#include <cassert>
#include <bitset>

#include "position4x4_cuda.h"
#include "position4x4_masks.h"

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

GameResult has_4_to_game_result_0_spaces_remain(bool has_4)
{
    switch (has_4)
    {
    case true:
        return GameResult::GameWin;
    case false:
        return GameResult::GameLose;
    }

    std::cout << "impossible" << std::endl;
    throw std::runtime_error("impossible");
}

__global__ void cuda_have_4_in_a_row(uint32_t *a, bool *b, int N)
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
    uint32_t *a;
    bool *b;

    // Allocation memory for these pointers
    cudaMallocManaged(&a, N * sizeof(uint32_t));
    cudaMallocManaged(&b, N * sizeof(bool));

    // Initialize vectors
    for (int i = 0; i < N; i++)
    {
        a[i] = rand() % 100;
    }

    // --------------------- CUDA ---------------------
    auto t1 = high_resolution_clock::now();

    // Call CUDA kernel
    cuda_have_4_in_a_row<<<GRID_SIZE(N), BLOCK_SIZE>>>(a, b, N);

    // Wait for all previous operations before using values
    // We need this because we don't get the implicit synchronization of
    // cudaMemcpy like in the original example
    cudaDeviceSynchronize();

    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n";
    std::cout << std::endl;

    // --------------------- CPU ---------------------
    auto t1 = high_resolution_clock::now();

    // TODO

    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n";
    std::cout << std::endl;

    // --------------------- CUDA == CPU ---------------------
    //  Verify the result on the CPU
    for (int i = 0; i < N; i++)
    {
        assert(b[i] == int_has_4_in_a_row(a[i]));
    }

    // Free unified memory (same as memory allocated with cudaMalloc)
    cudaFree(a);
    cudaFree(b);

    std::cout << "CUDA test completed successfully!\n";
}

std::map<Position4x4, GameResult> cuda_solve_0_spaces_remain()
{
    // Array size of 2^16 (65536 elements)
    const int N = 1 << 16;

    // Declare unified memory pointers
    uint32_t *a;
    bool *b;

    // Allocation memory for these pointers
    cudaMallocManaged(&a, N * sizeof(uint32_t));
    cudaMallocManaged(&b, N * sizeof(bool));

    // Initialize vectors
    for (uint32_t i = 0; i < N; i++)
    {
        a[i] = (i & 0b0000000000000001) |
               0b00000000000000000000000000000010 | (i & 0b0000000000000010) << 1 |
               0b00000000000000000000000000001000 | (i & 0b0000000000000100) << 2 |
               0b00000000000000000000000000100000 | (i & 0b0000000000001000) << 3 |
               0b00000000000000000000000010000000 | (i & 0b0000000000010000) << 4 |
               0b00000000000000000000001000000000 | (i & 0b0000000000100000) << 5 |
               0b00000000000000000000100000000000 | (i & 0b0000000001000000) << 6 |
               0b00000000000000000010000000000000 | (i & 0b0000000010000000) << 7 |
               0b00000000000000001000000000000000 | (i & 0b0000000100000000) << 8 |
               0b00000000000000100000000000000000 | (i & 0b0000001000000000) << 9 |
               0b00000000000010000000000000000000 | (i & 0b0000010000000000) << 10 |
               0b00000000001000000000000000000000 | (i & 0b0000100000000000) << 11 |
               0b00000000100000000000000000000000 | (i & 0b0001000000000000) << 12 |
               0b00000010000000000000000000000000 | (i & 0b0010000000000000) << 13 |
               0b00001000000000000000000000000000 | (i & 0b0100000000000000) << 14 |
               0b00100000000000000000000000000000 | (i & 0b1000000000000000) << 15 |
               0b10000000000000000000000000000000;
    }

    // Call CUDA kernel
    cuda_have_4_in_a_row<<<GRID_SIZE(N), BLOCK_SIZE>>>(a, b, N);

    // Wait for all previous operations before using values
    // We need this because we don't get the implicit synchronization of
    // cudaMemcpy like in the original example
    cudaDeviceSynchronize();

    // Insert to map
    auto result = std::map<Position4x4, GameResult>();

    for (uint32_t i = 0; i < N; i++)
    {
        uint32_t ai = a[i];

        auto position = Position4x4(Player::Order, (const char(&)[4])(ai), 0);
        auto game_result = has_4_to_game_result_0_spaces_remain(b[i]);

        result.insert(std::pair<Position4x4, GameResult>(
            position, game_result));
    }

    // Free unified memory (same as memory allocated with cudaMalloc)
    cudaFree(a);
    cudaFree(b);

    return result;
}
