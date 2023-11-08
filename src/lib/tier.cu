#include "tier.h"

Tier::SolveResult Tier::solve(SolveBy solve_by) {
  switch (solve_by) {
  case SolveBy::CPU:
    solve_by_cpu(position_hash_to_rv, next_tier->position_hash_to_rv);
    break;
  case SolveBy::GPU:
    // solve_by_gpu(position_hash_to_rv, next_tier->position_hash_to_rv);
    break;
  }

  return SolveResult::Success;
}

void solve_by_cpu(RecursiveValue *position_hash_to_rv,
                  RecursiveValue *child_position_hash_to_rv) {
  position_hash_to_rv[0] = RecursiveValue::Tie;
}

__global__ void solve_by_gpu(RecursiveValue *position_hash_to_rv,
                             RecursiveValue *child_position_hash_to_rv) {
  // position_hash_to_rv[0] = RecursiveValue::Tie;
}
