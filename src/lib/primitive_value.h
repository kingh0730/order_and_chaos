#pragma once

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
  std::string format() const;
};
