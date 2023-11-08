#pragma once

#include "position.h"
#include "recursive_value.h"

class Tier {
private:
  unsigned int num_empty_spaces;
  Tier *next_tier;
  RecursiveValue *position_hash_to_rv;

public:
  Tier(unsigned int num_empty_spaces, Tier *next_tier)
      : num_empty_spaces(num_empty_spaces), next_tier(next_tier) {
    position_hash_to_rv = (RecursiveValue *)malloc(0); // FIXME
  }

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

  std::string format() const;
};

void solve_by_cpu(RecursiveValue *position_hash_to_rv,
                  RecursiveValue *child_position_hash_to_rv);

__global__ void solve_by_gpu(RecursiveValue *position_hash_to_rv,
                             RecursiveValue *child_position_hash_to_rv);
