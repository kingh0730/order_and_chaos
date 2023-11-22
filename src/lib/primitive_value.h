#pragma once

#include "recursive_value.h"
#include "utils.h"
#include <string>

class PrimitiveValue {
public:
  enum _PrimitiveValueType {
    NotPrimitive = 0,
    Win = 1,
    Lose = 2,
    Tie = 3,
  };

private:
  _PrimitiveValueType pv;

public:
  CUDA_CALLABLE PrimitiveValue(_PrimitiveValueType pv) : pv(pv) {}

  CUDA_CALLABLE bool operator==(const PrimitiveValue &o) const {
    return pv == o.pv;
  }
  CUDA_CALLABLE bool operator!=(const PrimitiveValue &o) const {
    return !(*this == o);
  }

  CUDA_CALLABLE RecursiveValue to_recursive_value() const;

  std::string format() const;
};
