#pragma once

#include "utils.h"
#include <stdint.h>
#include <string>

class RecursiveValue {
public:
  enum _RecursiveValueType : uint8_t {
    Undetermined = 0,
    Win = 1,
    Lose = 2,
    Tie = 3,
    Draw = 4,
  };

private:
  _RecursiveValueType rv;

public:
  CUDA_CALLABLE RecursiveValue(_RecursiveValueType rv) : rv(rv) {}
  CUDA_CALLABLE RecursiveValue() : RecursiveValue(Undetermined) {}

  CUDA_CALLABLE bool operator==(const RecursiveValue &o) const {
    return rv == o.rv;
  }
  CUDA_CALLABLE bool operator!=(const RecursiveValue &o) const {
    return !(*this == o);
  }

  std::string format() const;
};
