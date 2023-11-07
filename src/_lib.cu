__global__ void vectorAdd(int *a, int *b, int *c, int N) {
  // Calculate global thread thread ID
  int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

  // Boundary check
  if (tid < N) {
    c[tid] = a[tid] + b[tid];
  }
}
