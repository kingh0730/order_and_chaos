#pragma once

#include "position.h"
#include "recursive_value.h"
#include <map>

class Tier {
private:
  Tier *next_tier;
  std::map<Position, RecursiveValue> position_to_rv;

public:
  Tier(int num_empty_spaces, Tier *next_tier) : next_tier(next_tier) {
    position_to_rv = std::map<Position, RecursiveValue>();
  }
};
