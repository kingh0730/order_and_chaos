#pragma once

__global__ bool cuda_int_has_4_in_a_row(const uint32_t &chars);
__global__ void cuda_have_4_in_a_row(int *a, bool *b, int N);

void test_cuda_have_4_in_a_row();
