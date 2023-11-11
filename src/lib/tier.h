#pragma once

#include "position.h"
#include "recursive_value.h"

class Tier {
private:
  unsigned int num_empty_spaces;
  unsigned long long num_positions;
  Tier *next_tier;
  RecursiveValue *position_hash_to_rv;
  bool solved;

public:
  Tier(unsigned int num_empty_spaces, Tier *next_tier);

  ~Tier() {
    delete position_hash_to_rv;
    return;
  }

  enum class SolveBy {
    CPU,
    GPU,
  };

  enum class SolveResult {
    Success,
    Error,
  };

  SolveResult solve(SolveBy solve_by);
  bool is_solved() const { return solved; }

  void set_next_tier_null() { next_tier = nullptr; }

  bool operator==(const Tier &other) const;

  std::string format() const;
};

void solve_by_cpu(RecursiveValue *position_hash_to_rv,
                  RecursiveValue *child_position_hash_to_rv,
                  unsigned long long num_positions);

__global__ void solve_by_gpu(RecursiveValue *position_hash_to_rv,
                             RecursiveValue *child_position_hash_to_rv,
                             unsigned long long num_positions);
