#pragma once

// If flag TTN_N is set then use it
// Otherwise, use default value 3

#ifndef TTT_N
#define TTT_N 4
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1 << 10
#endif

#ifndef GRID_SIZE
#define GRID_SIZE(N, BLOCK_SIZE) ((N) + (BLOCK_SIZE)-1) / (BLOCK_SIZE)
#endif
