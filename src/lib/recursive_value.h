#pragma once

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
  RecursiveValue(_RecursiveValueType rv) : rv(rv) {}
  RecursiveValue() : RecursiveValue(Undetermined) {}

  bool operator==(const RecursiveValue &other) const { return rv == other.rv; }

  std::string format() const;
};
