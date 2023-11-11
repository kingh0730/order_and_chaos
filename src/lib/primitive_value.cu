#include <iostream>
#include <string>

#include "primitive_value.h"

RecursiveValue PrimitiveValue::to_recursive_value() const {
  switch (pv) {
  case PrimitiveValue::NotPrimitive:
    return RecursiveValue::Undetermined;

  case PrimitiveValue::Win:
    return RecursiveValue::Win;

  case PrimitiveValue::Lose:
    return RecursiveValue::Lose;

  case PrimitiveValue::Tie:
    return RecursiveValue::Tie;

  default:
    std::cerr << "PrimitiveValue::Unknown(" + std::to_string(pv) + ")";
    throw std::invalid_argument("unknown PrimitiveValue type");
  }
}

std::string PrimitiveValue::format() const {
  switch (pv) {
  case PrimitiveValue::NotPrimitive:
    return "PrimitiveValue::NotPrimitive";

  case PrimitiveValue::Win:
    return "PrimitiveValue::Win";

  case PrimitiveValue::Lose:
    return "PrimitiveValue::Lose";

  case PrimitiveValue::Tie:
    return "PrimitiveValue::Tie";

  default:
    std::cerr << "PrimitiveValue::Unknown(" + std::to_string(pv) + ")";
    throw std::invalid_argument("unknown PrimitiveValue type");
  }
}
