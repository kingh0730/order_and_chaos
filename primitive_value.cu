#include <iostream>
#include <string>

#include "primitive_value.h"

std::string format_primitive_value(const PrimitiveValue &pv)
{
    switch (pv)
    {
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
