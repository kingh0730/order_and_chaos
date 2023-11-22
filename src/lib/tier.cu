#include "tier.h"
#include <iostream>
#include <sstream>
#include <stdint.h>

Tier::Tier(unsigned int num_empty_spaces, Tier *next_tier)
    : num_empty_spaces(num_empty_spaces), next_tier(next_tier) {

  // ! Use max_id >= num_positions
  num_positions = Position::max_id(num_empty_spaces) + 1;

  position_hash_to_rv = new RecursiveValue[num_positions];
  solved = false;
}

bool Tier::operator==(const Tier &other) const {
  if (num_empty_spaces != other.num_empty_spaces) {
    std::cout << "num_empty_spaces: " << num_empty_spaces;
    std::cout << " != " << other.num_empty_spaces << std::endl;
    return false;
  }
  if (num_positions != other.num_positions) {
    std::cout << "num_positions: " << num_positions;
    std::cout << " != " << other.num_positions << std::endl;
    return false;
  }
  if (solved != other.solved) {
    std::cout << "solved: " << solved;
    std::cout << " != " << other.solved << std::endl;
    return false;
  }
  // if (next_tier != other.next_tier) {
  //   return false;
  // }

  for (unsigned long long i = 0; i < num_positions; i++) {
    if (position_hash_to_rv[i] != other.position_hash_to_rv[i]) {
      std::cout << "position_hash_to_rv[" << i << "]: ";
      std::cout << position_hash_to_rv[i].format();
      std::cout << " != " << other.position_hash_to_rv[i].format() << std::endl;
      return false;
    }
  }

  return true;
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
    solve_by_cpu(position_hash_to_rv, child_position_hash_to_rv,
                 num_empty_spaces, num_positions);
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
        d_position_hash_to_rv, d_child_position_hash_to_rv, num_empty_spaces,
        num_positions);

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
                  RecursiveValue *child_position_hash_to_rv,
                  unsigned int num_empty_spaces,
                  unsigned long long num_positions) {
  for (unsigned long long id = 0; id < num_positions; id++) {
    Position position = Position(id, num_empty_spaces);

    auto pv = position.primitive_value();
    if (pv != PrimitiveValue::NotPrimitive) {
      position_hash_to_rv[id] = pv.to_recursive_value();
      continue;
    }

    Position *children;
    unsigned int num_children = position.children(children);

    for (unsigned int i = 0; i < num_children; i++) {
      unsigned long long child_id = children[i].id();
      if (child_position_hash_to_rv[child_id] == RecursiveValue::Lose) {
        position_hash_to_rv[id] = RecursiveValue::Win;
        break;
      }
    }

    if (position_hash_to_rv[id] == RecursiveValue::Win) {
      delete[] children;
      continue;
    }

    for (unsigned int i = 0; i < num_children; i++) {
      unsigned long long child_id = children[i].id();
      if (child_position_hash_to_rv[child_id] == RecursiveValue::Tie) {
        position_hash_to_rv[id] = RecursiveValue::Tie;
        break;
      }
    }

    if (position_hash_to_rv[id] == RecursiveValue::Tie) {
      delete[] children;
      continue;
    }

    position_hash_to_rv[id] = RecursiveValue::Lose;
    delete[] children;
    continue;
  }
}

__global__ void solve_by_gpu(RecursiveValue *position_hash_to_rv,
                             RecursiveValue *child_position_hash_to_rv,
                             unsigned int num_empty_spaces,
                             unsigned long long num_positions) {
  unsigned long long id = (blockDim.x * blockIdx.x) + threadIdx.x;

  Position position = Position(id, num_empty_spaces);

  auto cast_rv = (uint8_t *)position_hash_to_rv;
  cast_rv[id] = 3;
}

CUDA_CALLABLE void solve_common(RecursiveValue *position_hash_to_rv,
                                RecursiveValue *child_position_hash_to_rv,
                                unsigned int num_empty_spaces,
                                unsigned long long num_positions,
                                unsigned long long id) {
  Position position = Position(id, num_empty_spaces);

  auto pv = position.primitive_value();
  if (pv != PrimitiveValue::NotPrimitive) {
    position_hash_to_rv[id] = pv.to_recursive_value();
    return;
  }

  Position *children;
  unsigned int num_children = position.children(children);

  for (unsigned int i = 0; i < num_children; i++) {
    unsigned long long child_id = children[i].id();
    if (child_position_hash_to_rv[child_id] == RecursiveValue::Lose) {
      position_hash_to_rv[id] = RecursiveValue::Win;
      break;
    }
  }

  if (position_hash_to_rv[id] == RecursiveValue::Win) {
    delete[] children;
    return;
  }

  for (unsigned int i = 0; i < num_children; i++) {
    unsigned long long child_id = children[i].id();
    if (child_position_hash_to_rv[child_id] == RecursiveValue::Tie) {
      position_hash_to_rv[id] = RecursiveValue::Tie;
      break;
    }
  }

  if (position_hash_to_rv[id] == RecursiveValue::Tie) {
    delete[] children;
    return;
  }

  position_hash_to_rv[id] = RecursiveValue::Lose;
  delete[] children;
  return;
}
