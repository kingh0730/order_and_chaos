#pragma once

#include "recursive_value.h"
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
  PrimitiveValue(_PrimitiveValueType pv) : pv(pv) {}

  bool operator==(const PrimitiveValue &o) const { return pv == o.pv; }
  bool operator!=(const PrimitiveValue &o) const { return !(*this == o); }

  RecursiveValue to_recursive_value() const;

  std::string format() const;
};
