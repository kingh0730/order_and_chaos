#include "recursive_value.h"

#include <iostream>

std::string RecursiveValue::format() const {
  switch (rv) {
  case RecursiveValue::Undetermined:
    return "RecursiveValue::Undetermined";

  case RecursiveValue::Win:
    return "RecursiveValue::Win";

  case RecursiveValue::Lose:
    return "RecursiveValue::Lose";

  case RecursiveValue::Tie:
    return "RecursiveValue::Tie";

  default:
    std::cerr << "RecursiveValue::Unknown(" + std::to_string(rv) + ")";
    throw std::invalid_argument("unknown RecursiveValue type");
  }
}
