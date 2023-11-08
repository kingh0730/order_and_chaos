#include "tier.h"

void solve_by_cpu(RecursiveValue *position_hash_to_rv,
                  RecursiveValue *child_position_hash_to_rv) {
  position_hash_to_rv[0] = RecursiveValue::Tie;
}

__global__ void solve_by_gpu(RecursiveValue *position_hash_to_rv,
                             RecursiveValue *child_position_hash_to_rv) {
  position_hash_to_rv[0] = RecursiveValue::Tie;
}
