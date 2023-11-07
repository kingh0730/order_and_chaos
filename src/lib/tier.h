#pragma once

#include "position.h"
#include <map>

class Tier {
private:
  Tier *next_tier;
  std::map<Position, PrimitiveValue> position_to_pv;

public:
  Tier(int num_empty_spaces, Tier *next_tier) : next_tier(next_tier) {
    position_to_pv = std::map<Position, PrimitiveValue>();
  }
};
