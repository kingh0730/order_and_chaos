#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

// __global__ means this is called from the CPU, and runs on the GPU
__global__ void vectorAdd(
    const int *__restrict a,
    const int *__restrict b,
    int *__restrict c,
    int N
) {
  // Calculate global thread ID
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

  // Boundary check
  if (tid < N) c[tid] = a[tid] + b[tid];
}

// Check vector add result
void verify_result(
    std::vector<int> &a,
    std::vector<int> &b,
    std::vector<int> &c
) {
  for (int i = 0; i < a.size(); i++) {
    assert(c[i] == a[i] + b[i]);
  }
}
