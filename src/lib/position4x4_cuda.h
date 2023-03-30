#pragma once

__global__ void cuda_have_4_in_a_row(int *a, bool *b, int N);

void test_cuda_have_4_in_a_row();
