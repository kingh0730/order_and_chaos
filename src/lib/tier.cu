#include "tier.h"
#include <sstream>

Tier::Tier(unsigned int num_empty_spaces, Tier *next_tier)
    : num_empty_spaces(num_empty_spaces), next_tier(next_tier) {

  // ! Use max_id instead of num_positions
  num_positions = Position::max_id(num_empty_spaces);

  position_hash_to_rv = new RecursiveValue[num_positions];
  solved = false;
}

std::string Tier::format() const {
  std::stringstream ss;
  ss << "Tier:\n";
  ss << "  num_empty_spaces: " << num_empty_spaces << "\n";
  ss << "  num_positions: " << num_positions << "\n";
  ss << "  solved: " << (solved ? "True" : "False") << "\n";
  return ss.str();
}

Tier::SolveResult Tier::solve(SolveBy solve_by) {

  auto child_position_hash_to_rv =
      next_tier ? next_tier->position_hash_to_rv : nullptr;

  auto child_num_positions = next_tier ? next_tier->num_positions : 0;

  switch (solve_by) {

  case SolveBy::CPU:
    solve_by_cpu(position_hash_to_rv, child_position_hash_to_rv);
    break;

  case SolveBy::GPU:
    RecursiveValue *d_position_hash_to_rv, *d_child_position_hash_to_rv;

    unsigned long long position_hash_to_rv_size =
        sizeof(RecursiveValue) * num_positions;
    unsigned long long child_position_hash_to_rv_size =
        sizeof(RecursiveValue) * child_num_positions;
    cudaMalloc(&d_position_hash_to_rv, position_hash_to_rv_size);
    cudaMemcpy(d_position_hash_to_rv, position_hash_to_rv,
               position_hash_to_rv_size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_child_position_hash_to_rv, child_position_hash_to_rv_size);
    cudaMemcpy(d_child_position_hash_to_rv, child_position_hash_to_rv,
               child_position_hash_to_rv_size, cudaMemcpyHostToDevice);

    solve_by_gpu<<<GRID_SIZE(num_positions, BLOCK_SIZE), BLOCK_SIZE>>>(
        d_position_hash_to_rv, d_child_position_hash_to_rv);

    cudaMemcpy(position_hash_to_rv, d_position_hash_to_rv,
               position_hash_to_rv_size, cudaMemcpyDeviceToHost);

    cudaFree(d_position_hash_to_rv);
    cudaFree(d_child_position_hash_to_rv);
    break;
  }

  solved = true;
  return SolveResult::Success;
}

void solve_by_cpu(RecursiveValue *position_hash_to_rv,
                  RecursiveValue *child_position_hash_to_rv) {
  position_hash_to_rv[0] = RecursiveValue::Tie;
}

__global__ void solve_by_gpu(RecursiveValue *position_hash_to_rv,
                             RecursiveValue *child_position_hash_to_rv) {
  position_hash_to_rv[0] = RecursiveValue::Tie;
}
