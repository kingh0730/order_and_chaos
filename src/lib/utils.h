#pragma once

#ifndef __CUDACC__
#define __global__
#endif

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

unsigned long long factorial(unsigned int n);
unsigned long long combination(unsigned int n, unsigned int k);
