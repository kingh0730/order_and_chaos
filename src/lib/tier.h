#pragma once

#include "position.h"
#include "recursive_value.h"
#include <map>

class Tier {
private:
  unsigned int num_empty_spaces;
  Tier *next_tier;
  std::map<Position, RecursiveValue> position_to_rv;

public:
  Tier(unsigned int num_empty_spaces, Tier *next_tier)
      : num_empty_spaces(num_empty_spaces), next_tier(next_tier) {
    position_to_rv = std::map<Position, RecursiveValue>();
  }
};
