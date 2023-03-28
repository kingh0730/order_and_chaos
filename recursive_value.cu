#include <iostream>

#include "recursive_value.h"

GameResult to_game_result(const PrimitiveValue &pv)
{
    switch (pv)
    {
    case PrimitiveValue::NotPrimitive:
        return GameResult::Undecided;

    case PrimitiveValue::Win:
        return GameResult::Win;

    case PrimitiveValue::Lose:
        return GameResult::Lose;

    case PrimitiveValue::Tie:
        return GameResult::Tie;

    default:
        std::cerr << "Unknown primitive value: " << pv << std::endl;
        throw std::invalid_argument("Unknown primitive value");
    }
}
