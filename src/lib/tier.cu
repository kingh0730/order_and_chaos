#include "tier.h"
#include <chrono>
#include <iostream>
#include <sstream>
#include <stdint.h>

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::system_clock;

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

static RecursiveValue *last_d_position_hash_to_rv = nullptr;

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

    system_clock::time_point t1;
    system_clock::time_point t2;

    t1 = high_resolution_clock::now();
    cudaMalloc(&d_position_hash_to_rv, position_hash_to_rv_size);
    t2 = high_resolution_clock::now();
    std::cout << "\tcudaMalloc(..., position_hash_to_rv_size): ";
    std::cout << duration_cast<milliseconds>(t2 - t1).count() << "ms"
              << " from time " << t1.time_since_epoch().count() << " to "
              << t2.time_since_epoch().count() << std::endl;

    // cudaMemcpy(d_position_hash_to_rv, position_hash_to_rv,
    //            position_hash_to_rv_size, cudaMemcpyHostToDevice);

    // t1 = high_resolution_clock::now();
    // cudaMalloc(&d_child_position_hash_to_rv, child_position_hash_to_rv_size);
    // t2 = high_resolution_clock::now();
    // std::cout << "\tcudaMalloc(..., child_position_hash_to_rv_size): ";
    // std::cout << duration_cast<milliseconds>(t2 - t1).count() << "ms"
    //           << " from time " << t1.time_since_epoch().count() << " to "
    //           << t2.time_since_epoch().count() << std::endl;

    // t1 = high_resolution_clock::now();
    // cudaMemcpy(d_child_position_hash_to_rv, child_position_hash_to_rv,
    //            child_position_hash_to_rv_size, cudaMemcpyHostToDevice);
    // t2 = high_resolution_clock::now();
    // std::cout << "\tcudaMemcpy(..., child, cudaMemcpyHostToDevice): ";
    // std::cout << duration_cast<milliseconds>(t2 - t1).count() << "ms"
    //           << " from time " << t1.time_since_epoch().count() << " to "
    //           << t2.time_since_epoch().count() << std::endl;

    d_child_position_hash_to_rv = last_d_position_hash_to_rv;

    t1 = high_resolution_clock::now();
    solve_by_gpu<<<GRID_SIZE(num_positions, BLOCK_SIZE), BLOCK_SIZE>>>(
        d_position_hash_to_rv, d_child_position_hash_to_rv, num_empty_spaces,
        num_positions);
    t2 = high_resolution_clock::now();
    std::cout << "\tsolve_by_gpu: ";
    std::cout << duration_cast<milliseconds>(t2 - t1).count() << "ms"
              << " from time " << t1.time_since_epoch().count() << " to "
              << t2.time_since_epoch().count() << std::endl;

    last_d_position_hash_to_rv = d_position_hash_to_rv;

    t1 = high_resolution_clock::now();
    cudaMemcpy(position_hash_to_rv, d_position_hash_to_rv,
               position_hash_to_rv_size, cudaMemcpyDeviceToHost);
    t2 = high_resolution_clock::now();
    std::cout << "\tcudaMemcpy(..., self, cudaMemcpyDeviceToHost): ";
    std::cout << duration_cast<milliseconds>(t2 - t1).count() << "ms"
              << " from time " << t1.time_since_epoch().count() << " to "
              << t2.time_since_epoch().count() << std::endl;

    // t1 = high_resolution_clock::now();
    // cudaFree(d_position_hash_to_rv);
    // t2 = high_resolution_clock::now();
    // std::cout << "\tcudaFree(d_position_hash_to_rv): ";
    // std::cout << duration_cast<milliseconds>(t2 - t1).count() << "ms"
    //           << " from time " << t1.time_since_epoch().count() << " to "
    //           << t2.time_since_epoch().count() << std::endl;

    t1 = high_resolution_clock::now();
    cudaFree(d_child_position_hash_to_rv);
    t2 = high_resolution_clock::now();
    std::cout << "\tcudaFree(d_child_position_hash_to_rv): ";
    std::cout << duration_cast<milliseconds>(t2 - t1).count() << "ms"
              << " from time " << t1.time_since_epoch().count() << " to "
              << t2.time_since_epoch().count() << std::endl;

    break;
  }

  solved = true;
  return SolveResult::Success;
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

void solve_by_cpu(RecursiveValue *position_hash_to_rv,
                  RecursiveValue *child_position_hash_to_rv,
                  unsigned int num_empty_spaces,
                  unsigned long long num_positions) {
  for (unsigned long long id = 0; id < num_positions; id++) {
    solve_common(position_hash_to_rv, child_position_hash_to_rv,
                 num_empty_spaces, num_positions, id);
  }
}

__global__ void solve_by_gpu(RecursiveValue *position_hash_to_rv,
                             RecursiveValue *child_position_hash_to_rv,
                             unsigned int num_empty_spaces,
                             unsigned long long num_positions) {
  unsigned long long id = (blockDim.x * blockIdx.x) + threadIdx.x;

  if (id >= num_positions) {
    return;
  }

  solve_common(position_hash_to_rv, child_position_hash_to_rv, num_empty_spaces,
               num_positions, id);

  // auto cast_rv = (int8_t *)position_hash_to_rv;
  // cast_rv[id] = 3;
}

std::string Tier::format(const SolveBy &solve_by) {
  switch (solve_by) {
  case SolveBy::CPU:
    return "Tier::SolveBy::CPU";
  case SolveBy::GPU:
    return "Tier::SolveBy::GPU";
  default:
    std::cerr << "Tier::SolveBy::Unknown" << std::endl;
    throw std::invalid_argument("Unknown Tier::SolveBy");
  }
}
